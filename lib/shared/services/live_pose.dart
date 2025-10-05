
// Copyright
// Live per-frame 2D pipeline for Flutter using ONNX models (YOLOv8 + RTMPose).
// - Letterbox to 640x640 (pad=114) → YOLO → unletterbox → crop→256x192 → RTMPose (SimCC) → map back to image
// - Saves per-frame 2D results to a JSONL file in the temporary directory
// - No autodetection: parameters are explicit via LivePoseOptions
//
// This file avoids hard dependencies on your OrtSession type. Instead, you pass
// two inference callbacks: one for YOLO and one for RTMPose, so existing code stays intact.
//
// Usage sketch (pseudocode):
//   final live = await LivePose.create(options: LivePoseOptions());
//   await live.startRecording();
//   // per frame (RGB bytes, w, h):
//   final frame = await live.processFrameRgb(rgbBytes, width, height);
//   // draw frame.kptsCoco and frame.bbox
//   final pathJsonl = await live.stopRecording(); // path to coco_2d.jsonl
//
// NOTE: For production throughput, prefer native YUV->RGB conversion and resizing.
// This pure-Dart version is correct but not optimized for 30–60 fps workloads.

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:path_provider/path_provider.dart';

/// Options strictly mirroring the Python pipeline.
class LivePoseOptions {
  // YOLO
  final int yoloInput; // 640
  final List<int> padColor; // [114,114,114]
  final double personConf;
  final double personIou;
  final int personClassId; // 0
  /// 'normalized' or 'pixels' for YOLO head output units
  final String yoloUnits;
  /// 'letterbox' or 'original' for YOLO head coordinate domain
  final String yoloCoords;

  // RTMPose
  final int rtmH; // 256
  final int rtmW; // 192
  /// 'rgb_255' | 'bgr_255' | 'rgb_ms' | 'bgr_ms'
  final String rtmPreproc;
  final double simccRatio; // 2.0
  final int simccX; // 384
  final int simccY; // 512

  // Crop
  final double cropScale; // 1.25

  const LivePoseOptions({
    this.yoloInput = 640,
    this.padColor = const [114,114,114],
    this.personConf = 0.25,
    this.personIou = 0.45,
    this.personClassId = 0,
    this.yoloUnits = 'normalized',
    this.yoloCoords = 'letterbox',
    this.rtmH = 256,
    this.rtmW = 192,
    this.rtmPreproc = 'rgb_255',
    this.simccRatio = 2.0,
    this.simccX = 384,
    this.simccY = 512,
    this.cropScale = 1.25,
  });
}

/// YOLO inference callback signature.
/// Must return the raw detection tensor as flattened Float32List of shape [1,84,N].
typedef YoloRun = Future<Float32List> Function(Float32List chwInput /*[1,3,640,640]*/);

/// RTMPose inference callback signature.
/// Must return {'simcc_x': Float32List[1,17,Wx], 'simcc_y': Float32List[1,17,Hy]}.
typedef RtmRun = Future<Map<String, Float32List>> Function(Float32List chwInput /*[1,3,256,192]*/);

/// Per-frame output for overlay and logging.
class LivePoseFrame {
  final int t;
  /// Bounding box [x1,y1,x2,y2] in image pixels.
  final List<double> bbox;
  /// COCO-17 keypoints [[x,y,conf] * 17], image pixel coords.
  final List<List<double>> kptsCoco;
  /// Mean keypoint confidence for quick QA.
  final double confMean;

  LivePoseFrame(this.t, this.bbox, this.kptsCoco, this.confMean);
}

class LivePose {
  LivePose._(this.options, this._yoloRun, this._rtmRun);
  final LivePoseOptions options;
  final YoloRun _yoloRun;
  final RtmRun _rtmRun;

  late IOSink _jsonlSink;
  late String _jsonlPath;
  int _t = 0;
  List<double>? _prevBbox; // prev detection for stability

  static Future<LivePose> create({
    required LivePoseOptions options,
    required YoloRun yoloRun,
    required RtmRun rtmRun,
  }) async {
    return LivePose._(options, yoloRun, rtmRun);
  }

  Future<void> startRecording() async {
    final dir = await getTemporaryDirectory();
    _jsonlPath = '${dir.path}/coco_2d.jsonl';
    final file = File(_jsonlPath);
    if (await file.exists()) await file.delete();
    _jsonlSink = file.openWrite(mode: FileMode.writeOnlyAppend);
    _t = 0;
    _prevBbox = null;
  }

  Future<String> stopRecording() async {
    await _jsonlSink.flush();
    await _jsonlSink.close();
    return _jsonlPath;
  }

  /// Process one RGB frame (packed RGB uint8, HxWx3).
  Future<LivePoseFrame> processFrameRgb(Uint8List rgb, int width, int height) async {
    // 1) Letterbox to [640,640], keep r, dw, dh
    final lb = _letterbox(rgb, width, height, options.yoloInput, options.yoloInput, pad: options.padColor);

    // 2) YOLO input [1,3,640,640] float32 normalized depending on preproc (RGB/255)
    final yoloIn = _rgbToCHWFloat(lb.image, lb.canvasW, lb.canvasH, divideBy255: true);
    final yoloOut = await _yoloRun(yoloIn); // flattened [1,84,N]
    final int n = yoloOut.length ~/ 84; // assume [1,84,N]
    if (n == 0) {
      // no detections was returned; fallback
      final bbox = _fallbackBox(width, height);
      final crop = _cropToAspect(rgb, width, height, bbox, options.rtmH, options.rtmW, scale: options.cropScale);
      final rtmIn = _rgbToCHWFloat(crop.image, options.rtmW, options.rtmH, divideBy255: options.rtmPreproc.endsWith('255'));
      final simcc = await _rtmRun(rtmIn);
      final coords = _simccDecode(simcc['simcc_x']!, simcc['simcc_y']!, options.simccRatio, options.simccX, options.simccY);
      final coco = _coordsToImage(coords, crop.rect, options.rtmW, options.rtmH);
      final confMean = _mean(coco.map((e) => e[2]).toList());
      final frame = LivePoseFrame(_t, bbox, coco, confMean);
      await _writeJsonl(_t, bbox, confMean, coco);
      _t += 1;
      return frame;
    }

    // 3) Decode [1,84,N] → [N,84]
    final preds = _transpose84NToN84(yoloOut, n);

    // 4) Compute classes & scores
    final int C = 80; // COCO-80
    final scores = Float32List(n);
    final ids = Int32List(n);
    for (int i = 0; i < n; i++) {
      // logits → sigmoid for class scores
      double bestScore = -1.0;
      int bestId = -1;
      for (int c = 0; c < C; c++) {
        final v = _sigmoid(preds[i][4 + c]);
        if (v > bestScore) { bestScore = v; bestId = c; }
      }
      scores[i] = bestScore;
      ids[i] = bestId;
    }

    // 5) Keep person class and score threshold
    final keepPerson = <int>[];
    for (int i = 0; i < n; i++) {
      if (ids[i] == options.personClassId && scores[i] >= options.personConf) keepPerson.add(i);
    }

    // 6) Boxes (cx,cy,w,h) → xyxy in letterbox domain
    List<List<double>> boxesLb = [];
    List<double> scoresPerson = [];
    for (final i in keepPerson) {
      final cx = preds[i][0], cy = preds[i][1], w = preds[i][2], h = preds[i][3];
      double x1 = cx - w / 2.0;
      double y1 = cy - h / 2.0;
      double x2 = cx + w / 2.0;
      double y2 = cy + h / 2.0;
      boxesLb.add([x1, y1, x2, y2]);
      scoresPerson.add(scores[i]);
    }

    if (boxesLb.isEmpty) {
      // fallback to previous/center
      final bbox = _prevBbox ?? _fallbackBox(width, height);
      final crop = _cropToAspect(rgb, width, height, bbox, options.rtmH, options.rtmW, scale: options.cropScale);
      final rtmIn = _rgbToCHWFloat(crop.image, options.rtmW, options.rtmH, divideBy255: options.rtmPreproc.endsWith('255'));
      final simcc = await _rtmRun(rtmIn);
      final coords = _simccDecode(simcc['simcc_x']!, simcc['simcc_y']!, options.simccRatio, options.simccX, options.simccY);
      final coco = _coordsToImage(coords, crop.rect, options.rtmW, options.rtmH);
      final confMean = _mean(coco.map((e) => e[2]).toList());
      final frame = LivePoseFrame(_t, bbox, coco, confMean);
      await _writeJsonl(_t, bbox, confMean, coco);
      _t += 1;
      return frame;
    }

    // 7) If YOLO units are normalized, scale to pixels in letterbox
    if (options.yoloUnits == 'normalized') {
      for (var b in boxesLb) {
        b[0] *= options.yoloInput.toDouble();
        b[2] *= options.yoloInput.toDouble();
        b[1] *= options.yoloInput.toDouble();
        b[3] *= options.yoloInput.toDouble();
      }
    }

    // 8) Map from letterbox → original image, unless coords=='original'
    List<List<double>> boxesImg;
    if (options.yoloCoords == 'original') {
      boxesImg = boxesLb;
    } else {
      boxesImg = [];
      for (var b in boxesLb) {
        final x1 = (b[0] - lb.dw) / lb.r;
        final x2 = (b[2] - lb.dw) / lb.r;
        final y1 = (b[1] - lb.dh) / lb.r;
        final y2 = (b[3] - lb.dh) / lb.r;
        boxesImg.add([_clip(x1, 0, width - 1), _clip(y1, 0, height - 1),
                      _clip(x2, 0, width - 1), _clip(y2, 0, height - 1)]);
      }
    }

    // 9) NMS
    final keep = _nms(boxesImg, scoresPerson, options.personIou, 50);
    if (keep.isEmpty) {
      final bbox = _prevBbox ?? _fallbackBox(width, height);
      final crop = _cropToAspect(rgb, width, height, bbox, options.rtmH, options.rtmW, scale: options.cropScale);
      final rtmIn = _rgbToCHWFloat(crop.image, options.rtmW, options.rtmH, divideBy255: options.rtmPreproc.endsWith('255'));
      final simcc = await _rtmRun(rtmIn);
      final coords = _simccDecode(simcc['simcc_x']!, simcc['simcc_y']!, options.simccRatio, options.simccX, options.simccY);
      final coco = _coordsToImage(coords, crop.rect, options.rtmW, options.rtmH);
      final confMean = _mean(coco.map((e) => e[2]).toList());
      final frame = LivePoseFrame(_t, bbox, coco, confMean);
      await _writeJsonl(_t, bbox, confMean, coco);
      _t += 1;
      return frame;
    }

    final best = keep.first;
    final bbox = boxesImg[best];
    _prevBbox = bbox;

    // 10) Crop→256x192 + RTMPose
    final crop = _cropToAspect(rgb, width, height, bbox, options.rtmH, options.rtmW, scale: options.cropScale);
    final preDiv255 = options.rtmPreproc.endsWith('255');
    final rtmIn = _rgbToCHWFloat(crop.image, options.rtmW, options.rtmH, divideBy255: preDiv255);
    final simcc = await _rtmRun(rtmIn);

    // 11) SimCC decode and map back to image
    final coords = _simccDecode(simcc['simcc_x']!, simcc['simcc_y']!, options.simccRatio, options.simccX, options.simccY);
    final coco = _coordsToImage(coords, crop.rect, options.rtmW, options.rtmH);
    final confMean = _mean(coco.map((e) => e[2]).toList());

    // 12) Save JSONL (normalized bbox like Python)
    await _writeJsonl(_t, bbox, confMean, coco, imgW: width, imgH: height);

    final frame = LivePoseFrame(_t, bbox, coco, confMean);
    _t += 1;
    return frame;
  }

  // ----------------- Internals -----------------

  double _clip(double v, double lo, double hi) => v < lo ? lo : (v > hi ? hi : v);

  Future<void> _writeJsonl(int t, List<double> bbox, double score, List<List<double>> coco,
      {int? imgW, int? imgH}) async {
    final w = (imgW ?? 1).toDouble();
    final h = (imgH ?? 1).toDouble();
    final line = {
      "t": t,
      "bbox": [bbox[0]/w, bbox[1]/h, bbox[2]/w, bbox[3]/h],
      "score": score,
      "yolo_output_units": options.yoloUnits,
      "yolo_output_coords": options.yoloCoords,
      "kpt_coco": coco,
    };
    _jsonlSink.writeln(jsonEncode(line));
  }

  List<List<double>> _coordsToImage(List<List<double>> coordsIn, _Rect rect, int inW, int inH) {
    final rx = rect.x, ry = rect.y, rw = rect.w, rh = rect.h;
    final out = <List<double>>[];
    for (final p in coordsIn) {
      final x = rx + p[0] * (rw / inW);
      final y = ry + p[1] * (rh / inH);
      out.add([x, y, p[2]]);
    }
    return out;
  }

  List<List<double>> _simccDecode(Float32List simccX, Float32List simccY, double splitRatio, int Wx, int Hy) {
    // Shapes: [1,17,Wx], [1,17,Hy]
    const K = 17;
    final px = _softmaxLast(simccX, K, Wx);
    final py = _softmaxLast(simccY, K, Hy);
    final out = <List<double>>[];
    for (int k = 0; k < K; k++) {
      int argx = 0; double maxx = -1;
      for (int i = 0; i < Wx; i++) {
        final v = px[k * Wx + i];
        if (v > maxx) { maxx = v; argx = i; }
      }
      int argy = 0; double maxy = -1;
      for (int j = 0; j < Hy; j++) {
        final v = py[k * Hy + j];
        if (v > maxy) { maxy = v; argy = j; }
      }
      final x = argx / splitRatio;
      final y = argy / splitRatio;
      final conf = math.sqrt(maxx * maxy);
      out.add([x, y, conf]);
    }
    return out;
  }

  Float32List _softmaxLast(Float32List a, int K, int W) {
    final out = Float32List(K * W);
    for (int k = 0; k < K; k++) {
      // slice a[k,:]
      double maxv = -1e30;
      for (int i = 0; i < W; i++) {
        final v = a[k * W + i];
        if (v > maxv) maxv = v;
      }
      double sum = 0.0;
      for (int i = 0; i < W; i++) {
        final e = math.exp((a[k * W + i] - maxv));
        out[k * W + i] = e.toDouble();
        sum += e;
      }
      final inv = 1.0 / sum;
      for (int i = 0; i < W; i++) {
        out[k * W + i] = (out[k * W + i] * inv).toDouble();
      }
    }
    return out;
  }

  double _sigmoid(double x) => 1.0 / (1.0 + math.exp(-x));

  List<int> _nms(List<List<double>> boxes, List<double> scores, double iouThr, int maxDet) {
    final idxs = List<int>.generate(boxes.length, (i) => i);
    idxs.sort((a,b) => scores[b].compareTo(scores[a]));
    final keep = <int>[];
    while (idxs.isNotEmpty && keep.length < maxDet) {
      final i = idxs.removeAt(0);
      keep.add(i);
      idxs.removeWhere((j) => _iou(boxes[i], boxes[j]) > iouThr);
    }
    return keep;
  }

  double _iou(List<double> a, List<double> b) {
    final x1 = math.max(a[0], b[0]);
    final y1 = math.max(a[1], b[1]);
    final x2 = math.min(a[2], b[2]);
    final y2 = math.min(a[3], b[3]);
    final inter = math.max(0.0, x2 - x1) * math.max(0.0, y2 - y1);
    final areaA = math.max(0.0, a[2] - a[0]) * math.max(0.0, a[3] - a[1]);
    final areaB = math.max(0.0, b[2] - b[0]) * math.max(0.0, b[3] - b[1]);
    final denom = areaA + areaB - inter + 1e-9;
    return inter / denom;
  }

  double _mean(List<double> v) {
    if (v.isEmpty) return 0.0;
    double s = 0.0;
    for (final x in v) s += x;
    return s / v.length;
  }

  List<List<double>> _transpose84NToN84(Float32List flat84N, int N) {
    // Input shape [1,84,N]; ONNX is row-major (last dim contiguous).
    // So flat84N is laid out as 84 blocks of size N: [84][N].
    final out = List<List<double>>.generate(N, (_) => List<double>.filled(84, 0.0));
    int idx = 0;
    for (int c = 0; c < 84; c++) {
      for (int i = 0; i < N; i++) {
        out[i][c] = flat84N[idx++];
      }
    }
    return out;
  }

  _LetterboxResult _letterbox(Uint8List rgb, int w, int h, int tw, int th, {List<int> pad = const [114,114,114]}) {
    final r = math.min(th / h, tw / w);
    final newW = (w * r).round();
    final newH = (h * r).round();
    final resized = _resizeRgbNearest(rgb, w, h, newW, newH);
    final dwf = (tw - newW) / 2.0;
    final dhf = (th - newH) / 2.0;
    final left = dwf.floor();
    final top  = dhf.floor();
    final canvas = Uint8List(tw * th * 3);
    // fill pad color
    for (int i = 0; i < canvas.length; i += 3) {
      canvas[i+0] = pad[0];
      canvas[i+1] = pad[1];
      canvas[i+2] = pad[2];
    }
    // blit resized into canvas at (left, top)
    for (int y = 0; y < newH; y++) {
      final srcRow = y * newW * 3;
      final dstRow = (top + y) * tw * 3 + left * 3;
      canvas.setRange(dstRow, dstRow + newW * 3, resized, srcRow);
    }
    return _LetterboxResult(
      image: canvas,
      canvasW: tw,
      canvasH: th,
      r: r,
      dw: left.toDouble(),
      dh: top.toDouble(),
    );
  }

  Uint8List _resizeRgbNearest(Uint8List src, int sw, int sh, int dw, int dh) {
    final out = Uint8List(dw * dh * 3);
    final xRatio = sw / dw;
    final yRatio = sh / dh;
    for (int y = 0; y < dh; y++) {
      final sy = (y * yRatio).floor().clamp(0, sh - 1);
      for (int x = 0; x < dw; x++) {
        final sx = (x * xRatio).floor().clamp(0, sw - 1);
        final sIdx = (sy * sw + sx) * 3;
        final dIdx = (y * dw + x) * 3;
        out[dIdx]   = src[sIdx];
        out[dIdx+1] = src[sIdx+1];
        out[dIdx+2] = src[sIdx+2];
      }
    }
    return out;
  }

  Float32List _rgbToCHWFloat(Uint8List rgb, int w, int h, {bool divideBy255 = true, List<double>? mean, List<double>? std}) {
    final scale = divideBy255 ? 1.0 / 255.0 : 1.0;
    final out = Float32List(1 * 3 * h * w);
    int p = 0;
    final offR = 0 * h * w;
    final offG = 1 * h * w;
    final offB = 2 * h * w;
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final i = (y * w + x) * 3;
        final r = rgb[i].toDouble();
        final g = rgb[i + 1].toDouble();
        final b = rgb[i + 2].toDouble();
        out[offR + p] = (r * scale).toDouble();
        out[offG + p] = (g * scale).toDouble();
        out[offB + p] = (b * scale).toDouble();
        p++;
      }
    }
    if (mean != null && std != null && mean.length == 3 && std.length == 3) {
      // apply (x - mean)/std channel-wise
      for (int i = 0; i < h * w; i++) {
        out[offR + i] = ((out[offR + i] * 255.0) - mean[0]) / std[0];
        out[offG + i] = ((out[offG + i] * 255.0) - mean[1]) / std[1];
        out[offB + i] = ((out[offB + i] * 255.0) - mean[2]) / std[2];
      }
    }
    return out;
  }

  _CropResult _cropToAspect(Uint8List rgb, int w, int h, List<double> boxXYXY, int outH, int outW, {double scale = 1.25}) {
    final x1 = boxXYXY[0], y1 = boxXYXY[1], x2 = boxXYXY[2], y2 = boxXYXY[3];
    final cx = (x1 + x2) / 2.0;
    final cy = (y1 + y2) / 2.0;
    double bw = (x2 - x1) * scale;
    double bh = (y2 - y1) * scale;

    final targetAr = outW / outH; // 192/256
    if (bw / bh > targetAr) bh = bw / targetAr;
    else bw = bh * targetAr;

    int rx1 = math.max(0, (cx - bw / 2.0).round());
    int ry1 = math.max(0, (cy - bh / 2.0).round());
    int rx2 = math.min(w - 1, (cx + bw / 2.0).round());
    int ry2 = math.min(h - 1, (cy + bh / 2.0).round());

    final rw = math.max(1, rx2 - rx1);
    final rh = math.max(1, ry2 - ry1);
    final crop = Uint8List(outW * outH * 3);
    // nearest resize into outW×outH
    for (int y = 0; y < outH; y++) {
      final sy = (ry1 + (y * rh / outH)).floor().clamp(0, h - 1);
      for (int x = 0; x < outW; x++) {
        final sx = (rx1 + (x * rw / outW)).floor().clamp(0, w - 1);
        final sIdx = (sy * w + sx) * 3;
        final dIdx = (y * outW + x) * 3;
        crop[dIdx]   = rgb[sIdx];
        crop[dIdx+1] = rgb[sIdx+1];
        crop[dIdx+2] = rgb[sIdx+2];
      }
    }
    return _CropResult(
      image: crop,
      rect: _Rect(rx1.toDouble(), ry1.toDouble(), rw.toDouble(), rh.toDouble()),
    );
  }

  List<double> _fallbackBox(int w, int h) {
    return [w * 0.25, h * 0.1, w * 0.75, h * 0.9];
  }
}

class _LetterboxResult {
  final Uint8List image;
  final int canvasW;
  final int canvasH;
  final double r;
  final double dw;
  final double dh;
  _LetterboxResult({required this.image, required this.canvasW, required this.canvasH, required this.r, required this.dw, required this.dh});
}

class _Rect {
  final double x, y, w, h;
  const _Rect(this.x, this.y, this.w, this.h);
}

class _CropResult {
  final Uint8List image;
  final _Rect rect;
  const _CropResult({required this.image, required this.rect});
}