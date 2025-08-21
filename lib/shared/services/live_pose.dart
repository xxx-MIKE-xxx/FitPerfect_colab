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

  int _frameIdx = 0;
  _Det? _roi;              // last person box in raw coords
  _LetterboxMeta? _lb;     // last letterbox info to map 640→raw

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
    final kpts640 = _decodeSimCC(rtmOuts, crop.meta); // 640 coords

    // 5) Map kpts back to raw image space and return as offsets
    final ptsRaw = kpts640.map((k) {
      final xRaw = (k[0] - lb.meta.padX) / lb.meta.scale;
      final yRaw = (k[1] - lb.meta.padY) / lb.meta.scale;
      return Offset(xRaw, yRaw);
    }).toList(growable: false);

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

    // Manual paste: compatible with image ^4.x
    for (int y = 0; y < resized.height; y++) {
      for (int x = 0; x < resized.width; x++) {
        final p = resized.getPixel(x, y);
        out.setPixelRgba(
          padX + x,
          padY + y,
          getRed(p),
          getGreen(p),
          getBlue(p),
          getAlpha(p),
        );
      }
    }

    return _LetterboxResult(
      square: out,
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

  _Det? _pickBestPerson(OrtValue out) {
    final data = out.value as List;
    final flat = _toFloat32(data);
    if (flat.isEmpty) return null;

    const stride = 84; // xywh + obj + 80 classes
    final n = flat.length ~/ stride;
    double bestScore = 0.0;
    _Det? best;

    for (int i = 0; i < n; i++) {
      final off = i * stride;
      final cx = flat[off + 0], cy = flat[off + 1];
      final w  = flat[off + 2], h  = flat[off + 3];
      final score = _sigmoid(flat[off + 4 + 0]); // person class
      if (score < 0.25) continue;

      final x1 = math.max(0.0, cx - w / 2);
      final y1 = math.max(0.0, cy - h / 2);
      final x2 = math.min(640.0, cx + w / 2);
      final y2 = math.min(640.0, cy + h / 2);
      if (x2 <= x1 || y2 <= y1) continue;

      if (score > bestScore) {
        bestScore = score;
        best = _Det(x1, y1, x2, y2, score);
      }
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
