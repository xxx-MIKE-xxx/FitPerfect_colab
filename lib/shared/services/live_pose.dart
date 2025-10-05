// lib/shared/services/live_pose.dart
//
// Live image-stream 2D pose engine for Option B workflow.
// - Starts camera image stream
// - Processes sampled frames with YOLO→RTMPose
// - Publishes overlay points for UI (List<Offset> of 17 keypoints)
// - Writes one JSONL row per processed frame (COCO-17: [x,y,conf])
//
// This version fixes compile errors you saw:
//  • No OrtSession type annotations (use `dynamic` via your ort_session.dart wrapper)
//  • Handles OrtSession.run outputs as Map or List (no plugin-specific tensor cast)
//  • Adds estimate2D(...) that accepts either CameraImage or img.Image and returns List<Offset>
//  • overlayStream now emits List<Offset> to match UI expectations
//  • image v4 API (img.fill, Pixel.r/g/b)
//  • No references to dx on Strings, no MapEntry misuse in this file
//
// Copyright (c) FitPerfect

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' show Offset; // for overlay points

import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';

import 'ort_session.dart';        // OrtManager.fromAsset / .run()
import 'yuv_converter.dart';      // yuv420ToImage(CameraImage)

/// Live 2D engine.
class LivePoseEngine {
  LivePoseEngine({int? frameStride, int? yoloEvery, this.roiMargin = 1.25})
      : frameStride = frameStride ?? yoloEvery ?? 3;

  /// Process every Nth frame (sampling). Default 3 if not specified.
  final int frameStride;

  /// Scale bbox a bit when cropping for RTM (e.g., 1.25).
  final double roiMargin;

  // ───────────── runtime state ─────────────
  bool _running = false;
  bool _busy = false;
  int _frameIndex = 0;

  // Where 2D JSONL is written during live session.
  IOSink? _jsonlSink;
  late String _jsonlPath;
  late String _sessionId;
  late Directory _sessionDir;

  // UI overlay streams
  final _overlayPtsCtl = StreamController<List<Offset>>.broadcast();
  Stream<List<Offset>> get overlayStream => _overlayPtsCtl.stream; // <- used by UI

  // If a consumer wants raw data, they can listen here (optional)
  final _overlayDataCtl = StreamController<Map<String, dynamic>>.broadcast();
  Stream<Map<String, dynamic>> get overlayDataStream => _overlayDataCtl.stream;

  // Live ORT sessions (no static types; use your wrapper)
  dynamic _yolo;
  dynamic _rtm;

  /// Starts live processing on the given camera controller.
  /// Returns the sessionId used for file paths.
  Future<String> start({
    required CameraController controller,
    bool writeToDocuments = true,
    String? sessionId,
    void Function(Object error)? onError,
  }) async {
    if (_running) return _sessionId;
    _running = true;

    // Resolve session folder.
    _sessionId = sessionId ?? DateTime.now().millisecondsSinceEpoch.toString();
    final Directory baseDir = writeToDocuments
        ? await getApplicationDocumentsDirectory()
        : await getTemporaryDirectory();
    _sessionDir = Directory('${baseDir.path}/FitPerfect/$_sessionId');
    if (!(await _sessionDir.exists())) {
      await _sessionDir.create(recursive: true);
    }
    _jsonlPath = '${_sessionDir.path}/coco_2d.jsonl';
    _jsonlSink = File(_jsonlPath).openWrite(mode: FileMode.append);

    // Init ONNX sessions once (keep in current isolate for now).
    _yolo ??= await OrtManager.fromAsset('assets/models/yolov8n.onnx');
    _rtm  ??= await OrtManager.fromAsset('assets/models/rtmpose-m_256x192.onnx');

    // Start image stream.
    _frameIndex = 0;
    await controller.startImageStream((CameraImage camImage) async {
      // Frame sampling and backpressure.
      if (!_running) return;
      _frameIndex++;
      if ((_frameIndex % frameStride) != 0) return;
      if (_busy) return;
      _busy = true;
      try {
        await _processFrame(camImage);
      } catch (e) {
        if (onError != null) onError(e);
      } finally {
        _busy = false;
      }
    });

    return _sessionId;
  }

  /// Stops the image stream and finalizes the JSONL. Returns the JSONL path.
  Future<String> stop({CameraController? controller}) async {
    if (!_running) return _jsonlPath;
    _running = false;
    try {
      if (controller != null && controller.value.isStreamingImages) {
        await controller.stopImageStream();
      }
    } catch (_) {}
    await _jsonlSink?.flush();
    await _jsonlSink?.close();
    _jsonlSink = null;
    return _jsonlPath;
  }

  Future<void> dispose() async {
    await _overlayPtsCtl.close();
    await _overlayDataCtl.close();
    try { await _yolo?.close(); } catch (_) {}
    try { await _rtm?.close(); } catch (_) {}
  }

  // ───────────────────────── optional: single-frame API ─────────────────────────
  /// Back-compat for callers that invoke per-frame 2D estimation manually.
  /// Accepts either a CameraImage or an RGB img.Image, returns 17 Offsets.
  Future<List<Offset>> estimate2D(dynamic frame) async {
    _yolo ??= await OrtManager.fromAsset('assets/models/yolov8n.onnx');
    _rtm  ??= await OrtManager.fromAsset('assets/models/rtmpose-m_256x192.onnx');

    final img.Image rgb = (frame is CameraImage) ? yuv420ToImage(frame) : (frame as img.Image);
    final lb = _letterbox(rgb, 640, 640, pad: const [114, 114, 114]);
    final Float32List yoloIn = _toCHWFloat32(lb.image);
    final yoloOut = await _yolo!.run({'images': yoloIn});
    final Float32List pred = _asFloat32(_pickOutput(yoloOut, key: 'output0'));

    final decode = _decodeYolo(pred, lbW: 640, lbH: 640);
    final det = _pickBestPerson(decode);
    if (det == null) return const [];

    final boxPx = _unletterbox(det['xyxy'], lb);
    final cropRes = _cropForRtm(rgb, boxPx, outH: 256, outW: 192, margin: roiMargin);

    final Float32List rtmIn = _toCHWFloat32(cropRes.image);
    final rtmOut = await _rtm!.run({'input': rtmIn});
    final Float32List simccX = _asFloat32(_pickOutput(rtmOut, key: 'simcc_x', index: 0));
    final Float32List simccY = _asFloat32(_pickOutput(rtmOut, key: 'simcc_y', index: 1));

    final kptsIn = _simccDecode(simccX, simccY, 17, 384, 512, splitRatio: 2.0);
    final pts = <Offset>[];
    for (final kp in kptsIn) {
      final x = cropRes.rx + kp[0] * (cropRes.rw / 192.0);
      final y = cropRes.ry + kp[1] * (cropRes.rh / 256.0);
      pts.add(Offset(x, y));
    }
    return pts;
  }

  // ───────────────────────── private helpers ─────────────────────────

  Future<void> _processFrame(CameraImage cam) async {
    // Convert to RGB image (stride-aware, supports BGRA8888 / YUV420 variants).
    final rgb = yuv420ToImage(cam);

    // Letterbox to 640×640 (pad=114), keep mapping params.
    final lb = _letterbox(rgb, 640, 640, pad: const [114, 114, 114]);

    // Prepare YOLO input [1,3,640,640], float32 RGB/255.
    final Float32List yoloIn = _toCHWFloat32(lb.image); // 3*640*640
    final yoloOut = await _yolo!.run({'images': yoloIn});
    final Float32List pred = _asFloat32(_pickOutput(yoloOut, key: 'output0'));

    // Decode (assumes [1,84,N] flattened → [84*N]).
    final decode = _decodeYolo(pred, lbW: 640, lbH: 640);

    // Optional NMS; for single-person we pick max-score person.
    final det = _pickBestPerson(decode);
    if (det == null) return;

    // Unletterbox → original image pixels.
    final boxPx = _unletterbox(det['xyxy'], lb);

    // Crop with ROI margin and resize to 256×192.
    final cropRes = _cropForRtm(rgb, boxPx, outH: 256, outW: 192, margin: roiMargin);

    // RTMPose input [1,3,256,192], rgb_255
    final Float32List rtmIn = _toCHWFloat32(cropRes.image);
    final rtmOut = await _rtm!.run({'input': rtmIn});
    final Float32List simccX = _asFloat32(_pickOutput(rtmOut, key: 'simcc_x', index: 0));
    final Float32List simccY = _asFloat32(_pickOutput(rtmOut, key: 'simcc_y', index: 1));

    final kptsIn = _simccDecode(simccX, simccY, 17, 384, 512, splitRatio: 2.0);

    // Build overlay points and data.
    final pts = <Offset>[];
    final kptsPx = <List<double>>[];
    for (final kp in kptsIn) {
      final x = cropRes.rx + kp[0] * (cropRes.rw / 192.0);
      final y = cropRes.ry + kp[1] * (cropRes.rh / 256.0);
      final c = kp[2];
      pts.add(Offset(x, y));
      kptsPx.add([x, y, c]);
    }

    // Publish overlays (points first for UI, raw data optional).
    _overlayPtsCtl.add(pts);
    _overlayDataCtl.add({
      'bbox': [boxPx[0], boxPx[1], boxPx[2], boxPx[3]],
      'kpts': kptsPx,
    });

    // Write JSONL line (normalized bbox, absolute kpts).
    final w = rgb.width.toDouble();
    final h = rgb.height.toDouble();
    final line = json.encode({
      't': _frameIndex,
      'bbox': [boxPx[0] / w, boxPx[1] / h, boxPx[2] / w, boxPx[3] / h],
      'score': det['score'],
      'kpt_coco': kptsPx,
      'yolo_output_units': 'normalized_640_letterbox',
      'yolo_output_coords': 'letterbox',
      'lb': {'r': lb.r, 'dw': lb.dw, 'dh': lb.dh},
    });
    _jsonlSink?.writeln(line);
  }

  // Convert img.Image (RGB888) to CHW float32 [0..1].
  Float32List _toCHWFloat32(img.Image im) {
    final h = im.height, w = im.width;
    final out = Float32List(3 * h * w);
    int i0 = 0, i1 = h * w, i2 = 2 * h * w;
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final px = im.getPixel(x, y);     // Pixel in image v4
        out[i0++] = px.r / 255.0;
        out[i1++] = px.g / 255.0;
        out[i2++] = px.b / 255.0;
      }
    }
    return out;
  }

  /// Decode YOLOv8 flattened output into a list of detections.
  /// Expect layout [1,84,N] flattened → iterate N anchors, each 84 values.
  List<Map<String, dynamic>> _decodeYolo(Float32List pred, {int lbW = 640, int lbH = 640}) {
    final res = <Map<String, dynamic>>[];
    const stride = 84;
    final N = pred.length ~/ stride;
    for (int i = 0; i < N; i++) {
      final off = i * stride;
      final cx = pred[off + 0];
      final cy = pred[off + 1];
      final w  = pred[off + 2];
      final h  = pred[off + 3];
      // Person class prob at index 4 (sigmoid)
      final score = 1.0 / (1.0 + math.exp(-(pred[off + 4])));
      if (score < 0.15) continue;
      final x1 = (cx - w / 2.0) * lbW;
      final y1 = (cy - h / 2.0) * lbH;
      final x2 = (cx + w / 2.0) * lbW;
      final y2 = (cy + h / 2.0) * lbH;
      res.add({'xyxy': [x1, y1, x2, y2], 'score': score});
    }
    return res;
  }

  Map<String, dynamic>? _pickBestPerson(List<Map<String, dynamic>> dets) {
    if (dets.isEmpty) return null;
    dets.sort((a, b) => (b['score'] as double).compareTo(a['score'] as double));
    return dets.first;
  }

  // Letterbox with pad color [r,g,b] using image v4 API.
  _LetterboxResult _letterbox(img.Image src, int outW, int outH, {required List<int> pad}) {
    final r = math.min(outW / src.width, outH / src.height);
    final newW = (src.width * r).round();
    final newH = (src.height * r).round();
    final dw = ((outW - newW) / 2.0);
    final dh = ((outH - newH) / 2.0);

    final canvas = img.Image(width: outW, height: outH);
    // Fill background with pad color (v4: use top-level img.fill with Color)
    img.fill(canvas, color: img.ColorRgb8(pad[0], pad[1], pad[2]));

    final resized = img.copyResize(src, width: newW, height: newH, interpolation: img.Interpolation.linear);
    img.compositeImage(canvas, resized, dstX: dw.round(), dstY: dh.round());
    return _LetterboxResult(canvas, r, dw, dh);
  }

  // Undo letterbox to original image coords.
  List<double> _unletterbox(List<double> xyxy, _LetterboxResult lb) {
    return [
      (xyxy[0] - lb.dw) / lb.r,
      (xyxy[1] - lb.dh) / lb.r,
      (xyxy[2] - lb.dw) / lb.r,
      (xyxy[3] - lb.dh) / lb.r,
    ];
  }

  // Crop ROI around bbox with margin and resize for RTM.
  _CropResult _cropForRtm(img.Image src, List<double> xyxy, {required int outH, required int outW, double margin = 1.25}) {
    final x1 = xyxy[0], y1 = xyxy[1], x2 = xyxy[2], y2 = xyxy[3];
    final cx = (x1 + x2) / 2.0;
    final cy = (y1 + y2) / 2.0;
    double w = (x2 - x1).abs();
    double h = (y2 - y1).abs();
    final scale = margin;
    w *= scale; h *= scale;

    // keep aspect 192x256
    final target = outW / outH; // 192/256
    double rw = w, rh = h;
    if ((w / h) > target) {
      rh = w / target;
    } else {
      rw = h * target;
    }

    double rx = cx - rw / 2.0;
    double ry = cy - rh / 2.0;
    rx = rx.clamp(0.0, src.width.toDouble());
    ry = ry.clamp(0.0, src.height.toDouble());
    rw = math.min(rw, src.width - rx);
    rh = math.min(rh, src.height - ry);

    final crop = img.copyCrop(src, x: rx.round(), y: ry.round(), width: rw.round(), height: rh.round());
    final resized = img.copyResize(crop, width: outW, height: outH, interpolation: img.Interpolation.linear);
    return _CropResult(resized, rx, ry, rw, rh);
  }

  /// Minimal SimCC decoder: argmax on each joint's x/y logits.
  /// simccX: [1,J,W], simccY: [1,J,H]; coords returned in input-space (W->192, H->256) via splitRatio.
  List<List<double>> _simccDecode(Float32List simccX, Float32List simccY, int J, int Wx, int Hy, {double splitRatio = 2.0}) {
    final out = <List<double>>[];
    // Expect [J,Wx] after removing batch dim 1 (contiguous packing).
    for (int j = 0; j < J; j++) {
      int offX = j * Wx;
      int offY = j * Hy;
      // argmax X
      int ix = 0; double vx = -1e30;
      for (int k = 0; k < Wx; k++) {
        final v = simccX[offX + k];
        if (v > vx) { vx = v; ix = k; }
      }
      // argmax Y
      int iy = 0; double vy = -1e30;
      for (int k = 0; k < Hy; k++) {
        final v = simccY[offY + k];
        if (v > vy) { vy = v; iy = k; }
      }
      final x = ix / splitRatio; // 384/2 -> 192
      final y = iy / splitRatio; // 512/2 -> 256
      // crude confidence as mean of peak logits
      final c = ((vx + vy) / 2.0);
      out.add([x.toDouble(), y.toDouble(), c]);
    }
    return out;
  }

  // Pick output tensor by key (Map) or index (List); returns dynamic.
  dynamic _pickOutput(dynamic out, {String? key, int index = 0}) {
    if (out is Map && key != null && out.containsKey(key)) return out[key];
    if (out is Map && out.isNotEmpty) return out.values.first;
    if (out is List && out.isNotEmpty) {
      if (index >= 0 && index < out.length) return out[index];
      return out.first;
    }
    return out; // already a tensor
  }

  // Convert various OrtValue/tensor shapes to Float32List.
  Float32List _asFloat32(dynamic v) {
    if (v is Float32List) return v;
    // Common plugin wrapper types expose .data or .value
    try {
      final d = (v as dynamic).data;
      if (d is Float32List) return d;
      if (d is List) return Float32List.fromList(d.map((e) => (e as num).toDouble()).toList());
    } catch (_) {}
    try {
      final d = (v as dynamic).value;
      if (d is Float32List) return d;
      if (d is List) return Float32List.fromList(d.map((e) => (e as num).toDouble()).toList());
    } catch (_) {}
    if (v is List) {
      // Assume flat numeric list
      return Float32List.fromList(v.map((e) => (e as num).toDouble()).toList());
    }
    throw StateError('Unsupported tensor type: ${v.runtimeType}');
  }
}

class _LetterboxResult {
  _LetterboxResult(this.image, this.r, this.dw, this.dh);
  final img.Image image;
  final double r;
  final double dw;
  final double dh;
}

class _CropResult {
  _CropResult(this.image, this.rx, this.ry, this.rw, this.rh);
  final img.Image image;
  final double rx, ry, rw, rh;
}
