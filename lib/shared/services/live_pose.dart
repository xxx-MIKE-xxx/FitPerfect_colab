// lib/shared/services/live_pose.dart
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';

import 'ort_session.dart';
import 'tensor_utils.dart';
import 'yuv_converter.dart';

class LivePoseEngine {
  LivePoseEngine({this.yoloEvery = 5, this.roiMargin = 1.25});

  final int yoloEvery;     // refresh detection every N frames
  final double roiMargin;  // crop expansion

  OrtSession? _yolo;
  OrtSession? _rtm;

  Float32List? _yoloInputChw; // reusable buffer for YOLO input (1×3×640×640)
  Float32List? _rtmInputChw;  // reusable buffer for RTMPose input (1×3×256×192)

  int _frameIdx = 0;
  _Det? _roi;              // last person box in raw coords
  _LetterboxMeta? _lb;     // last letterbox info to map 640→raw
  bool _printedRtmShapes = false; // one-time debug print
  bool _reportedOutputSpace = false; // one-time keypoint space log

  Future<void> _ensureModelsLoaded() async {
    _yolo ??= await OrtManager.fromAsset('assets/models/yolov8n.onnx');
    _rtm  ??= await OrtManager.fromAsset('assets/models/rtmpose-m_256x192.onnx');
  }

  /// Main entry: returns 17 keypoints in **raw camera space** (width×height).
  Future<List<Offset>?> estimate2D(CameraImage cam) async {
    await _ensureModelsLoaded();
    // 1) YUV → RGB
    final rgb = yuv420ToImage(cam); // width=cam.width, height=cam.height

    // 2) Letterbox → 640×640 for YOLO (keep meta to unletterbox)
    final lb = _letterbox640(rgb);
    _lb = lb.meta;

    // 3) YOLO refresh?
    if (_roi == null || (_frameIdx % yoloEvery == 0)) {
      final yoloIn = _prepYolo(lb.square); // (1,3,640,640)
      final yoloOuts = await _run(_yolo!, {'images': yoloIn});
      final det640 = _pickBestPerson(yoloOuts.first!);
      if (det640 == null) {
        _frameIdx++;
        return null; // no person
      }
      // Map det from 640-square back to raw frame coords
      _roi = _lbToRaw(det640, lb.meta);
    }

    // 4) Crop & RTMPose in 640-square space for consistency
    // Build det in 640 coords from current raw roi
    final det640Now = _rawToLb(_roi!, lb.meta, clamp: true, margin: roiMargin);

    final crop = _cropForRtm(lb.square, det640Now, outH: 256, outW: 192);
    final rtmIn = _prepRtm(crop.image);
    final rtmOuts = await _run(_rtm!, {'input': rtmIn});
    if (!_printedRtmShapes && kDebugMode) {
      final s0 = rtmOuts.isNotEmpty ? _shapeOf(rtmOuts[0]?.value) : [];
      final s1 = rtmOuts.length > 1 ? _shapeOf(rtmOuts[1]?.value) : [];
      debugPrint('RTM outputs shapes (live): $s0${s1.isNotEmpty ? " & $s1" : ""}');
      _printedRtmShapes = true;
    }
    if (rtmOuts.isEmpty) {
      _frameIdx++;
      return null;
    }

    // Auto-detect SimCC vs Heatmap decoder
    final kpts640 = _decodeAuto(rtmOuts, crop.meta); // 640 coords

    // 5) Map kpts back to raw image space and return as offsets
    final ptsRaw = kpts640.map((k) {
      final xRaw = (k[0] - lb.meta.padX) / lb.meta.scale;
      final yRaw = (k[1] - lb.meta.padY) / lb.meta.scale;
      return Offset(xRaw, yRaw);
    }).toList(growable: false);

    if (kDebugMode && !_reportedOutputSpace) {
      debugPrint('[LivePoseEngine] Keypoints decoded from RTM crop 256x192, '
          'YOLO letterbox 640x640 → returning raw-space ${cam.width}x${cam.height} offsets');
      _reportedOutputSpace = true;
    }

    _frameIdx++;
    return ptsRaw;
  }

  // ─────────────────────────── internals ───────────────────────────

  Future<List<OrtValue?>> _run(OrtSession s, Map<String, OrtValue> inputs) async {
    final opts = OrtRunOptions();
    final outs = await s.runAsync(opts, inputs);
    opts.release();
    for (final v in inputs.values) { v.release(); }
    return outs ?? const [];
  }

  // Letterbox to 640×640
  _LetterboxResult _letterbox640(img.Image src) {
    final w = src.width, h = src.height;
    final scale = 640.0 / math.max(w, h);
    final nw = (w * scale).round();
    final nh = (h * scale).round();

    final resized = img.copyResize(
      src,
      width: nw,
      height: nh,
      interpolation: img.Interpolation.average,
    );

    // Create 640×640 black canvas
    final out = img.Image(width: 640, height: 640);
    img.fill(out, color: img.ColorRgb8(0, 0, 0));

    // Center the resized image
    final padX = ((640 - nw) ~/ 2);
    final padY = ((640 - nh) ~/ 2);

    // Fast path: let image package do the blit in one go
    final composited = img.compositeImage(out, resized, dstX: padX, dstY: padY);

    return _LetterboxResult(
      square: composited,
      meta: _LetterboxMeta(
        srcW: w,
        srcH: h,
        scale: scale,
        padX: padX.toDouble(),
        padY: padY.toDouble(),
      ),
    );
  }

  OrtValue _prepYolo(img.Image im) {
    final w = im.width, h = im.height; // 640×640
    final plane = w * h;
    final needed = 3 * plane;
    if (_yoloInputChw == null || _yoloInputChw!.length != needed) {
      _yoloInputChw = Float32List(needed);
    }

    final chw = _yoloInputChw!;
    final bytes = im.getBytes(order: img.ChannelOrder.rgb);

    for (int i = 0, p = 0; p < plane; p++) {
      final r = bytes[i++] / 255.0;
      final g = bytes[i++] / 255.0;
      final b = bytes[i++] / 255.0;
      chw[p] = r;
      chw[plane + p] = g;
      chw[(plane << 1) + p] = b;
    }

    return OrtValueTensor.createTensorWithDataList(chw, [1, 3, h, w]);
  }

  // ───────────────── YOLO decode (robust to shape & scale) ─────────────────
  _Det? _pickBestPerson(OrtValue out) {
    final v = out.value;
    final shp = _shapeOf(v); // e.g., [1, 84, 8400] or [1, 8400, 84]
    if (shp.length != 3) {
      // Fallback: flatten assuming stride 84
      final flat = _toFloat32(v);
      if (flat.isEmpty || flat.length % 84 != 0) return null;
      return _scanYoloFlat(flat, stride: 84);
    }

    // Normalize into a uniform iterator of (cx,cy,w,h,obj,cls0) across proposals.
    // Case A: [1, 84, N]   -> channels-first
    // Case B: [1, N, 84]   -> channels-last
    final arr = _toList3D(v); // List[1][A][B]
    final A = arr[0].length;
    final B = arr[0][0].length;

    double bestScore = 0.0;
    _Det? best;

    if (A == 84) {
      // [1, 84, N]
      final N = B;
      for (int i = 0; i < N; i++) {
        final cx = arr[0][0][i];
        final cy = arr[0][1][i];
        final w  = arr[0][2][i];
        final h  = arr[0][3][i];
        final obj  = arr[0][4][i];
        final cls0 = arr[0][5][i]; // "person"
        final det = _mkDet(cx, cy, w, h, obj, cls0);
        if (det != null && det.score > bestScore) { bestScore = det.score; best = det; }
      }
    } else if (B == 84) {
      // [1, N, 84]
      final N = A;
      for (int i = 0; i < N; i++) {
        final row = arr[0][i];
        final cx = row[0];
        final cy = row[1];
        final w  = row[2];
        final h  = row[3];
        final obj  = row[4];
        final cls0 = row[5]; // "person"
        final det = _mkDet(cx, cy, w, h, obj, cls0);
        if (det != null && det.score > bestScore) { bestScore = det.score; best = det; }
      }
    } else {
      // Unexpected; fallback to flat
      final flat = _toFloat32(v);
      if (flat.isEmpty || flat.length % 84 != 0) return null;
      return _scanYoloFlat(flat, stride: 84);
    }

    if (kDebugMode && best != null) {
      debugPrint(
        'YOLO best det 640: '
        'x1=${(best.x1/640.0).toStringAsFixed(1)} '
        'y1=${(best.y1/640.0).toStringAsFixed(1)} '
        'x2=${(best.x2/640.0).toStringAsFixed(1)} '
        'y2=${(best.y2/640.0).toStringAsFixed(1)} '
        'score=${best.score.toStringAsFixed(3)}'
      );
    }
    return best;
  }

  // Build a detection in 640-space, auto-scaling if inputs are normalized.
  _Det? _mkDet(double cx, double cy, double w, double h, double obj, double cls0) {
    // If numbers look like 0..1, scale to 640.
    final bool normalized = (cx <= 1.5 && cy <= 1.5 && w <= 1.5 && h <= 1.5);
    final s = normalized ? 640.0 : 1.0;

    final cxp = cx * s, cyp = cy * s, wp = w * s, hp = h * s;

    if (wp <= 1 || hp <= 1) return null; // ignore degenerate boxes

    final x1 = (cxp - wp / 2).clamp(0.0, 640.0);
    final y1 = (cyp - hp / 2).clamp(0.0, 640.0);
    final x2 = (cxp + wp / 2).clamp(0.0, 640.0);
    final y2 = (cyp + hp / 2).clamp(0.0, 640.0);

    if (x2 <= x1 || y2 <= y1) return null;

    // Score: objectness * class(person) with sigmoid just in case
    final score = _sigmoid(obj) * _sigmoid(cls0);
    if (score < 0.25) return null;

    return _Det(x1, y1, x2, y2, score);
  }

  _Det? _scanYoloFlat(Float32List flat, {required int stride}) {
    double bestScore = 0.0;
    _Det? best;
    final n = flat.length ~/ stride;
    for (int i = 0; i < n; i++) {
      final off = i * stride;
      final cx = flat[off + 0];
      final cy = flat[off + 1];
      final w  = flat[off + 2];
      final h  = flat[off + 3];
      final obj  = flat[off + 4];
      final cls0 = flat[off + 5];
      final det = _mkDet(cx, cy, w, h, obj, cls0);
      if (det != null && det.score > bestScore) { bestScore = det.score; best = det; }
    }
    return best;
  }

  double _sigmoid(double x) => 1 / (1 + math.exp(-x));

  // Crop & resize (640 space) for RTMPose
  _CropResult _cropForRtm(img.Image im640, _Det det, {required int outH, required int outW}) {
    final x1 = det.x1.clamp(0.0, im640.width.toDouble());
    final y1 = det.y1.clamp(0.0, im640.height.toDouble());
    final x2 = det.x2.clamp(0.0, im640.width.toDouble());
    final y2 = det.y2.clamp(0.0, im640.height.toDouble());

    final crop = img.copyCrop(
      im640,
      x: x1.toInt(),
      y: y1.toInt(),
      width: (x2 - x1).toInt(),
      height: (y2 - y1).toInt(),
    );

    final resized = img.copyResize(
      crop, width: outW, height: outH, interpolation: img.Interpolation.average);

    final meta = _CropMeta(
      roiX: x1, roiY: y1,
      roiW: (x2 - x1), roiH: (y2 - y1),
      outW: outW, outH: outH,
    );
    return _CropResult(resized, meta);
  }

  OrtValue _prepRtm(img.Image patch) {
    const mean = [0.485, 0.456, 0.406];
    const std  = [0.229, 0.224, 0.225];
    final h = patch.height, w = patch.width; // 256×192
    final plane = w * h;
    final needed = 3 * plane;
    if (_rtmInputChw == null || _rtmInputChw!.length != needed) {
      _rtmInputChw = Float32List(needed);
    }

    final chw = _rtmInputChw!;
    final bytes = patch.getBytes(order: img.ChannelOrder.rgb);

    for (int i = 0, p = 0; p < plane; p++) {
      final r = bytes[i++] / 255.0;
      final g = bytes[i++] / 255.0;
      final b = bytes[i++] / 255.0;

      chw[p] = (r - mean[0]) / std[0];
      chw[plane + p] = (g - mean[1]) / std[1];
      chw[(plane << 1) + p] = (b - mean[2]) / std[2];
    }

    return OrtValueTensor.createTensorWithDataList(chw, [1, 3, h, w]);
  }

  // ---------- NEW: Auto-detect SimCC vs Heatmap and decode accordingly

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
        // Make sure first is X (smaller bins) and second is Y (larger bins)
        final xFirst = s0.isNotEmpty && s1.isNotEmpty && s0.last <= s1.last;
        final pair = xFirst ? outs : [outs[1], outs[0]];
        return _decodeSimCC(pair, meta);
      }
    }

    // Fallback: try heatmap on first output
    return _decodeHeatmap(outs[0]!, meta);
  }

  // Heatmap decoder: argmax per joint over H×W plane, then map from 256×192 crop back to 640 space.
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

      // Then to the 640×640 ROI coords
      final x = meta.roiX + (px / meta.outW) * meta.roiW;
      final y = meta.roiY + (py / meta.outH) * meta.roiH;

      kpts[j][0] = x;
      kpts[j][1] = y;
      kpts[j][2] = 1.0; // optional: set to 1 since we used argmax
    }
    return kpts;
  }

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
      final conf = math.min(1.0, math.max(0.0, (vx + vy) / 2.0));

      kpts[j][0] = x; // still in 640 space
      kpts[j][1] = y;
      kpts[j][2] = conf;
    }
    return kpts;
  }

  // Map det in 640 space → raw space
  _Det _lbToRaw(_Det d, _LetterboxMeta m) {
    final x1 = (d.x1 - m.padX) / m.scale;
    final y1 = (d.y1 - m.padY) / m.scale;
    final x2 = (d.x2 - m.padX) / m.scale;
    final y2 = (d.y2 - m.padY) / m.scale;
    return _Det(x1, y1, x2, y2, d.score);
  }

  // Map det in raw space → 640 space (optionally expand margin and clamp)
  _Det _rawToLb(_Det d, _LetterboxMeta m, {bool clamp = true, double margin = 1.0}) {
    final cx = (d.x1 + d.x2) * 0.5;
    final cy = (d.y1 + d.y2) * 0.5;
    final w  = (d.x2 - d.x1) * margin;
    final h  = (d.y2 - d.y1) * margin;

    double x1 = cx - w * 0.5;
    double y1 = cy - h * 0.5;
    double x2 = cx + w * 0.5;
    double y2 = cy + h * 0.5;

    // to 640
    x1 = x1 * m.scale + m.padX;
    y1 = y1 * m.scale + m.padY;
    x2 = x2 * m.scale + m.padX;
    y2 = y2 * m.scale + m.padY;

    if (clamp) {
      x1 = x1.clamp(0.0, 640.0);
      y1 = y1.clamp(0.0, 640.0);
      x2 = x2.clamp(0.0, 640.0);
      y2 = y2.clamp(0.0, 640.0);
    }
    return _Det(x1, y1, x2, y2, d.score);
  }

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
}

// ───────── data holders ─────────
class _Det { _Det(this.x1, this.y1, this.x2, this.y2, this.score);
  final double x1, y1, x2, y2, score; }

class _CropMeta {
  _CropMeta({
    required this.roiX, required this.roiY,
    required this.roiW, required this.roiH,
    required this.outW, required this.outH,
  });
  final double roiX, roiY, roiW, roiH;
  final int outW, outH;
}

class _CropResult { _CropResult(this.image, this.meta);
  final img.Image image; final _CropMeta meta; }

class _LetterboxMeta {
  _LetterboxMeta({
    required this.srcW, required this.srcH,
    required this.scale, required this.padX, required this.padY,
  });
  final int srcW, srcH;
  final double scale, padX, padY;
}
class _LetterboxResult {
  _LetterboxResult({required this.square, required this.meta});
  final img.Image square;
  final _LetterboxMeta meta;
}
