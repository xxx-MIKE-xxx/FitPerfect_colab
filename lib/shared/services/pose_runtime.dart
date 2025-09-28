// lib/shared/services/pose_runtime.dart
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';

import 'ort_session.dart';
import 'video_sampler.dart';
import 'tensor_utils.dart';

// lib/shared/services/pose_runtime.dart
class PosePipelineResult {
  PosePipelineResult({required this.kpts2d, required this.kpts3d});
  final List<List<List<double>>> kpts2d; // [T][17][3] (x,y,conf)
  final List<List<List<double>>> kpts3d; // [T][17][3] (x,y,z)

  Map<String, dynamic> toReport() => {
    'num_frames': kpts2d.length,
    'num_joints': 17,
    'kpts2d': kpts2d,
    'kpts3d': kpts3d,
  };
}


class PosePipeline {
  OrtSession? _yolo;
  OrtSession? _rtm;
  OrtSession? _mb;

  Future<void> _ensureModelsLoaded() async {
    _yolo ??= await OrtManager.fromAsset('assets/models/yolov8n.onnx');
    _rtm  ??= await OrtManager.fromAsset('assets/models/rtmpose-m_256x192.onnx');
    _mb   ??= await OrtManager.fromAsset('assets/models/motionbert_lite_81.onnx');
  }

  Future<PosePipelineResult> analyzeVideo(File video, {int mbWindow = 81}) async {
    await _ensureModelsLoaded();

    final frames = await VideoSampler.extract10FpsJpgs(video);

    final all2D = <List<List<double>>>[];
    for (final f in frames) {
      final jpg = await f.readAsBytes();

      final yoloIn  = _prepYolo(jpg);                   // (1,3,640,640)
      final yoloOut = await _run(_yolo!, {'images': yoloIn});
      final det     = _pickBestPerson(yoloOut.first!);  // may be null

      if (det == null) {
        all2D.add(List.generate(17, (_) => [0.0, 0.0, 0.0]));
        continue;
      }

      final crop = _cropForRtm(jpg, det, outH: 256, outW: 192);
      final rtmIn   = _prepRtm(crop.image);                 // (1,3,256,192)
      final rtmOuts = await _run(_rtm!, {'input': rtmIn});
      if (kDebugMode && all2D.isEmpty) {
        debugPrint('RTM outputs shapes (offline): '
            '${_shapeOf(rtmOuts.isNotEmpty ? rtmOuts[0]?.value : null)}'
            '${rtmOuts.length > 1 ? ' & ${_shapeOf(rtmOuts[1]?.value)}' : ''}');
      }
      final kpts = _decodeAuto(rtmOuts, crop.meta);

      all2D.add(kpts);
    }

    final T = math.min(mbWindow, all2D.length);
    final mbIn   = _toMbInput(all2D.take(T).toList());               // [1,T,17,3]
    final mbOuts = await _run(_mb!, {'input': mbIn});
    final out3d  = _toList4D(mbOuts.first!.value, dims: [1, T, 17, 3])[0]; // ← .value

    return PosePipelineResult(kpts2d: all2D.take(T).toList(), kpts3d: out3d);
  }

  Future<List<OrtValue?>> _run(OrtSession s, Map<String, OrtValue> inputs) async {
    final opts = OrtRunOptions();
    final outs = await s.runAsync(opts, inputs);
    opts.release();
    for (final v in inputs.values) { v.release(); }
    return outs ?? const [];
  }

  // ---------- YOLOv8 preprocessing

  OrtValue _prepYolo(Uint8List jpgBytes) {
    final im = img.decodeImage(jpgBytes)!; // expect 640x640 (letterboxed by sampler)
    final w = im.width, h = im.height;
    final chw = Float32List(1 * 3 * h * w);
    int i = 0;
    for (int c = 0; c < 3; c++) {
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          final p = im.getPixel(x, y);
          final r = getRed(p), g = getGreen(p), b = getBlue(p);
          final v = (c == 0 ? r : (c == 1 ? g : b)) / 255.0;
          chw[i++] = v;
        }
      }
    }
    return OrtValueTensor.createTensorWithDataList(chw, [1, 3, h, w]);
  }

  // Robust YOLO decode: supports [1,N,84] and [1,84,N], uses obj×class(person) scoring.
  _Det? _pickBestPerson(OrtValue out) {
    final v = out.value;
    if (v is! List || v.isEmpty) return null;
    if (v[0] is! List) return null;

    // v[0] can be [N,84] or [84,N]
    final l0 = v[0] as List;
    if (l0.isEmpty) return null;

    // Determine layout by inspecting first inner list length.
    bool isNx84;
    if (l0[0] is List) {
      final firstInner = l0[0] as List;
      isNx84 = firstInner.length == 84; // [1, N, 84]
    } else {
      return null;
    }

    // Build rows = [cx, cy, w, h, obj, c0..c79]
    final rows = <List<double>>[];
    if (isNx84) {
      // [1, N, 84]
      for (final r in (v[0] as List)) {
        rows.add((r as List).map((e) => (e as num).toDouble()).toList());
      }
    } else {
      // [1, 84, N] -> transpose to [N,84]
      final C = (v[0] as List).length; // 84
      final N = ((v[0] as List)[0] as List).length;
      for (int n = 0; n < N; n++) {
        final row = <double>[];
        for (int c = 0; c < C; c++) {
          row.add((((v[0] as List)[c] as List)[n] as num).toDouble());
        }
        rows.add(row);
      }
    }

    double bestScore = 0.0;
    _Det? best;
    for (final r in rows) {
      if (r.length < 84) continue;
      final cx = r[0], cy = r[1], w = r[2], h = r[3];

      // Proper YOLO scoring: sigmoid(obj) * sigmoid(class_person)
      final obj = 1.0 / (1.0 + math.exp(-r[4]));
      final clsPerson = 1.0 / (1.0 + math.exp(-r[5])); // COCO class 0 = person
      final score = obj * clsPerson;
      if (score < 0.10) continue; // relaxed threshold

      final x1 = (cx - w / 2).clamp(0.0, 640.0);
      final y1 = (cy - h / 2).clamp(0.0, 640.0);
      final x2 = (cx + w / 2).clamp(0.0, 640.0);
      final y2 = (cy + h / 2).clamp(0.0, 640.0);
      if (x2 <= x1 || y2 <= y1) continue;

      if (score > bestScore) {
        bestScore = score;
        best = _Det(x1, y1, x2, y2, score);
      }
    }
    return best;
  }

  double _sigmoid(double x) => 1 / (1 + math.exp(-x));

  // ---------- RTMPose cropping + preprocessing

  _CropResult _cropForRtm(Uint8List jpg, _Det det, {required int outH, required int outW}) {
    final im = img.decodeImage(jpg)!; // 640x640
    final x1 = det.x1.clamp(0.0, im.width.toDouble());
    final y1 = det.y1.clamp(0.0, im.height.toDouble());
    final x2 = det.x2.clamp(0.0, im.width.toDouble());
    final y2 = det.y2.clamp(0.0, im.height.toDouble());

    final w = (x2 - x1), h = (y2 - y1);
    const scale = 1.25;
    final cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
    final halfW = (w * scale) / 2, halfH = (h * scale) / 2;
    final rx1 = (cx - halfW).clamp(0.0, im.width.toDouble()).toInt();
    final ry1 = (cy - halfH).clamp(0.0, im.height.toDouble()).toInt();
    final rx2 = (cx + halfW).clamp(0.0, im.width.toDouble()).toInt();
    final ry2 = (cy + halfH).clamp(0.0, im.height.toDouble()).toInt();

    final crop = img.copyCrop(im, x: rx1, y: ry1, width: (rx2 - rx1), height: (ry2 - ry1));
    final resized = img.copyResize(
      crop, width: outW, height: outH, interpolation: img.Interpolation.average);

    final meta = _CropMeta(
      srcW: im.width, srcH: im.height,
      roiX: rx1.toDouble(), roiY: ry1.toDouble(),
      roiW: (rx2 - rx1).toDouble(), roiH: (ry2 - ry1).toDouble(),
      outW: outW, outH: outH,
    );
    return _CropResult(resized, meta);
  }

  OrtValue _prepRtm(img.Image patch) {
    const mean = [0.485, 0.456, 0.406];
    const std  = [0.229, 0.224, 0.225];

    final h = patch.height, w = patch.width; // 256x192
    final chw = Float32List(1 * 3 * h * w);
    int i = 0;
    for (int c = 0; c < 3; c++) {
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          final p = patch.getPixel(x, y);
          final r = getRed(p) / 255.0;
          final g = getGreen(p) / 255.0;
          final b = getBlue(p) / 255.0;
          final v = c == 0
              ? (r - mean[0]) / std[0]
              : c == 1
                  ? (g - mean[1]) / std[1]
                  : (b - mean[2]) / std[2];
          chw[i++] = v;
        }
      }
    }
    return OrtValueTensor.createTensorWithDataList(chw, [1, 3, h, w]);
  }

  // Decode SIMCC outputs: outs = [(1,17,384), (1,17,512)]
  List<List<double>> _decodeSimCC(List<OrtValue?> outs, _CropMeta meta) {
    final simccX = _toList3D(outs[0]!.value); // [1,17,384]
    final simccY = _toList3D(outs[1]!.value); // [1,17,512]
    const split = 2.0;

    final List<List<double>> kpts = List.generate(17, (_) => [0.0, 0.0, 0.0]);
    for (int j = 0; j < 17; j++) {
      final rowX = simccX[0][j];
      final rowY = simccY[0][j];

      int ix = 0, iy = 0;
      double vx = -1e9, vy = -1e9;
      for (int t = 0; t < rowX.length; t++) { if (rowX[t] > vx) { vx = rowX[t]; ix = t; } }
      for (int t = 0; t < rowY.length; t++) { if (rowY[t] > vy) { vy = rowY[t]; iy = t; } }

      final px = ix / split; // 0..192
      final py = iy / split; // 0..256

      final x = meta.roiX + (px / meta.outW) * meta.roiW;
      final y = meta.roiY + (py / meta.outH) * meta.roiH;
      final conf = math.min(1.0, math.max(0.0, (vx + vy) / 2.0)); // rough

      kpts[j][0] = x;
      kpts[j][1] = y;
      kpts[j][2] = conf;
    }
    return kpts;
  }

  // ---------- NEW: auto-detect SimCC vs Heatmap and decode accordingly

  // Inspect nested list tensor shape (for debug/branching)
  List<int> _shapeOf(dynamic v) {
    final dims = <int>[];
    dynamic a = v;
    while (a is List) {
      dims.add(a.length);
      a = a.isNotEmpty ? a[0] : null;
    }
    return dims;
  }

  // Unified decoder: chooses SimCC or Heatmap at runtime based on output shapes.
  List<List<double>> _decodeAuto(List<OrtValue?> outs, _CropMeta meta) {
    if (outs.isEmpty || outs[0] == null) {
      return List.generate(17, (_) => [0.0, 0.0, 0.0]);
    }

    final v0 = outs[0]!.value;
    final s0 = _shapeOf(v0);

    // Heatmap ONNX commonly returns [1, K, H, W]
    if (s0.length == 4) {
      return _decodeHeatmap(outs[0]!, meta);
    }

    // SimCC ONNX: two outputs [1, K, 384] & [1, K, 512] (order may vary)
    if (outs.length >= 2) {
      final s1 = _shapeOf(outs[1]!.value);
      final looksSimCC = s0.length == 3 && s1.length == 3 && s0[0] == 1 && s1[0] == 1;
      if (looksSimCC) {
        // Pick the smaller "bins" tensor as X (usually 384 < 512)
        final xFirst = s0.isNotEmpty && s1.isNotEmpty && s0.last <= s1.last;
        return _decodeSimCC(xFirst ? outs : [outs[1], outs[0]], meta);
      }
    }

    // Fallback: try heatmap on first output
    return _decodeHeatmap(outs[0]!, meta);
  }

  // Heatmap decoder: argmax per joint over H×W plane, then map from 256×192 crop back to full frame.
  List<List<double>> _decodeHeatmap(OrtValue heat, _CropMeta meta) {
    // heat.value is nested lists [1][K][H][W]
    final n1 = heat.value as List;            // [1]
    if (n1.isEmpty) return List.generate(17, (_) => [0.0, 0.0, 0.0]);
    final nk = n1[0] as List;                 // [K]
    final K = nk.length;

    // infer H, W from first joint
    final H = (nk[0] as List).length;
    final W = ((nk[0] as List)[0] as List).length;

    final List<List<double>> kpts = List.generate(17, (_) => [0.0, 0.0, 0.0]);
    for (int j = 0; j < (K < 17 ? K : 17); j++) {
      final plane = nk[j] as List;            // [H][W]
      double best = -1e9;
      int by = 0, bx = 0;
      for (int y = 0; y < H; y++) {
        final row = plane[y] as List;
        for (int x = 0; x < W; x++) {
          final v = (row[x] as num).toDouble();
          if (v > best) { best = v; by = y; bx = x; }
        }
      }

      // Back to 256×192 crop coords (center of the heatmap cell)
      final px = (bx + 0.5) * (meta.outW / W);
      final py = (by + 0.5) * (meta.outH / H);

      // Then to the 640×640 letterbox ROI in the original frame
      final x = meta.roiX + (px / meta.outW) * meta.roiW;
      final y = meta.roiY + (py / meta.outH) * meta.roiH;

      kpts[j][0] = x;
      kpts[j][1] = y;
      kpts[j][2] = 1.0; // optional: normalize best and set as confidence
    }
    return kpts;
  }

  OrtValue _toMbInput(List<List<List<double>>> k2d) {
    final T = k2d.length;
    final buf = Float32List(1 * T * 17 * 3);
    int i = 0;
    for (int t = 0; t < T; t++) {
      for (int j = 0; j < 17; j++) {
        buf[i++] = k2d[t][j][0].toDouble();
        buf[i++] = k2d[t][j][1].toDouble();
        buf[i++] = k2d[t][j][2].toDouble();
      }
    }
    return OrtValueTensor.createTensorWithDataList(buf, [1, T, 17, 3]);
  }

  // -------- list helpers

  Float32List _toFloat32(dynamic v) {
    if (v is Float32List) return v;
    if (v is List) {
      final flat = <double>[];
      void walk(dynamic a) {
        if (a is List) { for (final e in a) walk(e); }
        else if (a is num) flat.add(a.toDouble());
      }
      walk(v);
      return Float32List.fromList(flat);
    }
    return Float32List(0);
  }

  List<List<List<double>>> _toList3D(dynamic v) =>
      (v as List).map<List<List<double>>>((a) =>
        (a as List).map<List<double>>((b) =>
          (b as List).map<double>((c) => (c as num).toDouble()).toList()
        ).toList()
      ).toList();

  List<List<List<List<double>>>> _toList4D(dynamic v, {required List<int> dims}) {
    return (v as List).map<List<List<List<double>>>>((a) =>
      (a as List).map<List<List<double>>>((b) =>
        (b as List).map<List<double>>((c) =>
          (c as List).map<double>((d) => (d as num).toDouble()).toList()
        ).toList()
      ).toList()
    ).toList();
  }
}

class _Det { _Det(this.x1, this.y1, this.x2, this.y2, this.score);
  final double x1, y1, x2, y2, score; }

class _CropMeta {
  _CropMeta({
    required this.srcW, required this.srcH,
    required this.roiX, required this.roiY,
    required this.roiW, required this.roiH,
    required this.outW, required this.outH,
  });
  final int srcW, srcH, outW, outH;
  final double roiX, roiY, roiW, roiH;
}
class _CropResult { _CropResult(this.image, this.meta);
  final img.Image image; final _CropMeta meta; }
