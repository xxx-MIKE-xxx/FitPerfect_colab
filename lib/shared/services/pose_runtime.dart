// lib/shared/services/pose_runtime.dart
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';

import 'ort_session.dart';
import 'tensor_utils.dart';
import 'video_sampler.dart';

/// Callback emitted from the offline pipeline when 3D progress changes.
typedef PosePipelineProgressCallback = void Function(PosePipelineProgress progress);

enum PosePipelinePhase { preparing3d, running3d }

class PosePipelineProgress {
  const PosePipelineProgress._(this.phase, this.processed, this.total);

  final PosePipelinePhase phase;
  final int processed;
  final int total;

  double get fraction => total == 0 ? 0.0 : processed / total;

  factory PosePipelineProgress.preparing3d() =>
      const PosePipelineProgress._(PosePipelinePhase.preparing3d, 0, 0);

  factory PosePipelineProgress.running3d({required int processed, required int total}) =>
      PosePipelineProgress._(PosePipelinePhase.running3d, processed, total);
}

class PoseKeypoint2D {
  const PoseKeypoint2D({
    required this.id,
    required this.x,
    required this.y,
    this.c,
  });

  final int id;
  final double x;
  final double y;
  final double? c;

  Map<String, dynamic> toJson() => {
        'id': id,
        'x': x,
        'y': y,
        if (c != null) 'c': c,
      };
}

class Pose2DFrame {
  const Pose2DFrame({
    required this.frameIndex,
    required this.t,
    required this.keypoints,
    required this.imgW,
    required this.imgH,
  });

  final int frameIndex;
  final double t;
  final List<PoseKeypoint2D> keypoints;
  final int imgW;
  final int imgH;

  Map<String, dynamic> toJson() => {
        'frameIndex': frameIndex,
        't': double.parse(t.toStringAsFixed(3)),
        'keypoints': keypoints.map((k) => k.toJson()).toList(),
        'imgW': imgW,
        'imgH': imgH,
      };

  List<List<double>> toRawList() => keypoints
      .map((k) => [k.x, k.y, k.c ?? 0.0])
      .toList(growable: false);
}

class Pose2DSequence {
  const Pose2DSequence({
    required this.frames,
    required this.fps,
    required this.imageWidth,
    required this.imageHeight,
  });

  final List<Pose2DFrame> frames;
  final double fps;
  final int imageWidth;
  final int imageHeight;

  bool get isEmpty => frames.isEmpty;
}

class Pose3DWindow {
  const Pose3DWindow({
    required this.windowIndex,
    required this.frameStart,
    required this.stride,
    required this.T,
    required this.joints3D,
    required this.units,
    required this.rootJoint,
    required this.pelvisCentered,
    required this.notes,
  });

  final int windowIndex;
  final int frameStart;
  final int stride;
  final int T;
  final List<List<List<double>>> joints3D;
  final String units;
  final String rootJoint;
  final bool pelvisCentered;
  final String notes;

  Map<String, dynamic> toJson() => {
        'windowIndex': windowIndex,
        'frameStart': frameStart,
        'stride': stride,
        'T': T,
        'joints3D': joints3D,
        'units': units,
        'rootJoint': rootJoint,
        'pelvisCentered': pelvisCentered,
        'notes': notes,
      };
}

class Pose3DResult {
  const Pose3DResult({
    required this.windows,
    required this.sequence,
    required this.windowSize,
    required this.stride,
    required this.pelvisCentered,
  });

  final List<Pose3DWindow> windows;
  final List<List<List<double>>> sequence; // [T][17][3]
  final int windowSize;
  final int stride;
  final bool pelvisCentered;
}

class PosePipelineResult {
  const PosePipelineResult({required this.pose2d, required this.pose3d});

  final Pose2DSequence pose2d;
  final Pose3DResult pose3d;

  Map<String, dynamic> toReport() => {
        'num_frames': pose2d.frames.length,
        'num_joints': 17,
        'kpts2d': pose2d.frames.map((f) => f.toRawList()).toList(),
        'kpts3d': pose3d.sequence,
        'meta': {
          'fps': pose2d.fps,
          'window': pose3d.windowSize,
          'stride': pose3d.stride,
        }
      };
}

class MotionBertConfig {
  const MotionBertConfig({
    required this.assetPath,
    required this.window,
    required this.stride,
  });

  final String assetPath;
  final int window;
  final int stride;
}

class PosePipelineCancelled implements Exception {}

class PosePipeline {
  // --- Letterbox state for last YOLO input ---
  double _lbR = 1.0; // scale used (min(tw/w, th/h))
  double _lbDw = 0.0; // pad x
  double _lbDh = 0.0; // pad y
  int _lbW = 640;
  int _lbH = 640;
  int _origW = 0;
  int _origH = 0;
  OrtSession? _yolo;
  OrtSession? _rtm;
  OrtSession? _mb;
  MotionBertConfig? _mbConfig;

  Future<void> _ensure2DModelsLoaded() async {
    _yolo ??= await OrtManager.fromAsset('assets/models/yolov8n.onnx');
    _rtm ??= await OrtManager.fromAsset('assets/models/rtmpose-m_256x192.onnx');
  }

  Future<void> _ensureMotionBertLoaded() async {
    if (_mb != null) return;
    final configs = [
      const MotionBertConfig(
        assetPath: 'assets/models/motionbert_3d_243.onnx',
        window: 243,
        stride: 81,
      ),
      const MotionBertConfig(
        assetPath: 'assets/models/motionbert_lite_81.onnx',
        window: 81,
        stride: 27,
      ),
    ];

    for (final cfg in configs) {
      try {
        final session = await OrtManager.fromAsset(cfg.assetPath);
        _mb = session;
        _mbConfig = cfg;
        break;
      } catch (e) {
        if (kDebugMode) {
          debugPrint('[PosePipeline] Could not load ${cfg.assetPath}: $e');
        }
        continue;
      }
    }

    if (_mb == null || _mbConfig == null) {
      throw Exception('MotionBERT model not found. Provide motionbert_3d_243.onnx or motionbert_lite_81.onnx.');
    }
  }

  MotionBertConfig get motionBertConfig {
    final cfg = _mbConfig;
    if (cfg == null) {
      throw StateError('MotionBERT not initialized yet');
    }
    return cfg;
  }

  Future<PosePipelineResult> analyzeVideo(
    File video, {
    PosePipelineProgressCallback? onProgress,
    bool Function()? shouldAbort,
  }) async {
    final seq = await extract2D(
      video,
      shouldAbort: shouldAbort,
    );
    final res3d = await estimate3D(
      seq,
      onProgress: onProgress,
      shouldAbort: shouldAbort,
    );
    return PosePipelineResult(pose2d: seq, pose3d: res3d);
  }

  Future<Pose2DSequence> extract2D(
    File video, {
    double targetFps = 10.0,
    bool Function()? shouldAbort,
  }) async {
    await _ensure2DModelsLoaded();

    final frames = await VideoSampler.extract10FpsJpgs(video);
    final results = <Pose2DFrame>[];

    for (int i = 0; i < frames.length; i++) {
      if (shouldAbort?.call() == true) throw PosePipelineCancelled();

      final file = frames[i];
      final jpg = await file.readAsBytes();
      final image = img.decodeImage(jpg);
      if (image == null) {
        results.add(_emptyFrame(i, targetFps));
        continue;
      }

      final yoloIn = _prepYolo(image);
      final yoloOuts = await _run(_yolo!, {'images': yoloIn});
      final det = yoloOuts.isEmpty ? null : _pickBestPerson(yoloOuts.first!);

      List<List<double>> h36m;
      if (det == null) {
        h36m = List.generate(17, (_) => [0.0, 0.0, 0.0]);
      } else {
        final crop = _cropForRtm(jpg, det, outH: 256, outW: 192);
        final rtmIn = _prepRtm(crop.image);
        final rtmOuts = await _run(_rtm!, {'input': rtmIn});
        final raw = _decodeAuto(rtmOuts, crop.meta);
        h36m = _remapToH36M(raw);
      }

      final frame = Pose2DFrame(
        frameIndex: i,
        t: i / targetFps,
        imgW: image.width,
        imgH: image.height,
        keypoints: List.generate(17, (j) {
          final kp = h36m[j];
          return PoseKeypoint2D(
            id: j,
            x: kp[0],
            y: kp[1],
            c: kp.length > 2 ? kp[2] : null,
          );
        }),
      );
      results.add(frame);
    }

    return Pose2DSequence(
      frames: results,
      fps: targetFps,
      imageWidth: results.isEmpty ? 0 : results.first.imgW,
      imageHeight: results.isEmpty ? 0 : results.first.imgH,
    );
  }

  Future<Pose3DResult> estimate3D(
    Pose2DSequence seq, {
    PosePipelineProgressCallback? onProgress,
    bool pelvisCentered = false,
    bool Function()? shouldAbort,
  }) async {
    await _ensureMotionBertLoaded();

    final frames = seq.frames;
    if (frames.isEmpty) {
      return Pose3DResult(
        windows: const [],
        sequence: const [],
        windowSize: motionBertConfig.window,
        stride: motionBertConfig.stride,
        pelvisCentered: pelvisCentered,
      );
    }

    onProgress?.call(PosePipelineProgress.preparing3d());

    final T = motionBertConfig.window;
    final stride = motionBertConfig.stride;
    final starts = _computeWindowStarts(frames.length, T, stride);

    final fused = List.generate(
      frames.length,
      (_) => List.generate(17, (_) => List<double>.filled(3, 0.0, growable: false),
          growable: false),
      growable: false,
    );
    final counts = List.generate(
      frames.length,
      (_) => List<int>.filled(17, 0, growable: false),
      growable: false,
    );

    final windows = <Pose3DWindow>[];
    int processed = 0;
    onProgress?.call(PosePipelineProgress.running3d(processed: processed, total: starts.length));

    for (final start in starts) {
      if (shouldAbort?.call() == true) throw PosePipelineCancelled();

      final window = _prepareWindow(frames, start, T);
      final input = _toMotionBertInput(window.frames);
      final outs = await _run(_mb!, {'input': input});
      if (outs.isEmpty) {
        processed++;
        onProgress?.call(PosePipelineProgress.running3d(processed: processed, total: starts.length));
        continue;
      }

      final out = outs.first!;
      final joints = _toList4D(out.value, dims: [1, T, 17, 3])[0];

      if (pelvisCentered) {
        for (int t = 0; t < joints.length; t++) {
          final pelvis = joints[t][0];
          for (int j = 0; j < joints[t].length; j++) {
            joints[t][j][0] -= pelvis[0];
            joints[t][j][1] -= pelvis[1];
            joints[t][j][2] -= pelvis[2];
          }
        }
      }

      windows.add(Pose3DWindow(
        windowIndex: windows.length,
        frameStart: start,
        stride: stride,
        T: T,
        joints3D: joints,
        units: 'normalized',
        rootJoint: 'pelvis',
        pelvisCentered: pelvisCentered,
        notes: 'inputs normalized: centered & scaled by min(W,H)/2',
      ));

      for (int t = 0; t < joints.length; t++) {
        final sourceIndex = window.indices[t];
        if (sourceIndex < 0 || sourceIndex >= fused.length) continue;
        for (int j = 0; j < 17; j++) {
          final joint = joints[t][j];
          fused[sourceIndex][j][0] += joint[0];
          fused[sourceIndex][j][1] += joint[1];
          fused[sourceIndex][j][2] += joint[2];
          counts[sourceIndex][j] += 1;
        }
      }

      processed++;
      onProgress?.call(
        PosePipelineProgress.running3d(processed: processed, total: starts.length),
      );
    }

    for (int i = 0; i < fused.length; i++) {
      for (int j = 0; j < fused[i].length; j++) {
        final c = counts[i][j];
        if (c == 0) continue;
        fused[i][j][0] /= c;
        fused[i][j][1] /= c;
        fused[i][j][2] /= c;
      }
    }

    return Pose3DResult(
      windows: windows,
      sequence: fused,
      windowSize: T,
      stride: stride,
      pelvisCentered: pelvisCentered,
    );
  }

  Pose2DFrame _emptyFrame(int index, double fps) => Pose2DFrame(
        frameIndex: index,
        t: index / fps,
        imgW: 640,
        imgH: 640,
        keypoints: List.generate(
          17,
          (j) => PoseKeypoint2D(id: j, x: 0.0, y: 0.0, c: 0.0),
        ),
      );

  Future<List<OrtValue?>> _run(OrtSession session, Map<String, OrtValue> inputs) async {
    final opts = OrtRunOptions();
    final outs = await session.runAsync(opts, inputs);
    opts.release();
    for (final v in inputs.values) {
      v.release();
    }
    return outs ?? const [];
  }

  
OrtValue _prepYolo(img.Image im) {
  // Store original dims
  _origW = im.width;
  _origH = im.height;

  // Letterbox to 640x640 with pad 114, keep scale/offsets
  const tw = 640, th = 640;
  final r = math.min(th / _origH, tw / _origW);
  final newW = (_origW * r).round();
  final newH = (_origH * r).round();
  final resized = img.copyResize(im, width: newW, height: newH, interpolation: img.Interpolation.nearest);
  final canvas = img.Image(width: tw, height: th);
  // fill with 114 gray
  img.fill(canvas, color: img.ColorRgb8(114, 114, 114));
  final left = ((tw - newW) / 2).floor();
  final top  = ((th - newH) / 2).floor();
  img.compositeImage(canvas, resized, dstX: left, dstY: top, dstW: newW, dstH: newH, blend: img.BlendMode.direct);

  // Save letterbox meta for unmapping
  _lbR = r;
  _lbDw = left.toDouble();
  _lbDh = top.toDouble();
  _lbW = tw;
  _lbH = th;

  // Export CHW float32 in RGB/255
  final w = canvas.width;
  final h = canvas.height;
  final plane = w * h;
  final buf = Float32List(3 * plane);
  final bytes = canvas.getBytes(order: img.ChannelOrder.rgb);
  for (int i = 0, p = 0; p < plane; p++) {
    final r8 = bytes[i++] / 255.0;
    final g8 = bytes[i++] / 255.0;
    final b8 = bytes[i++] / 255.0;
    buf[p] = r8;
    buf[plane + p] = g8;
    buf[(plane << 1) + p] = b8;
  }
  return OrtValueTensor.createTensorWithDataList(buf, [1, 3, h, w]);
}

_Det? _pickBestPerson(OrtValue out) {
  // Expect Ultralytics YOLOv8: [1, 84, N] (4 box + 80 classes).
  // Some exports may be [1, N, 84]; handle both.
  final v = out.value;
  if (v is! List || v.isEmpty) return null;
  final l0 = v[0];
  if (l0 is! List || l0.isEmpty) return null;

  // Convert to [N,84] double array
  List<List<double>> arr;
  // Try [84, N]
  bool parsed = false;
  try {
    final chans = (v[0] as List);
    final C = chans.length;
    final N = (chans[0] as List).length;
    if (C == 84) {
      arr = List.generate(N, (i) {
        final row = List<double>.filled(C, 0.0);
        for (int c = 0; c < C; c++) {
          row[c] = ((chans[c] as List)[i] as num).toDouble();
        }
        return row;
      });
      parsed = true;
    } else {
      arr = const [];
    }
  } catch (_) {
    arr = const [];
  }
  if (!parsed) {
    // Fallback: assume [N,84]
    final rows = (v[0] as List);
    arr = rows
        .map<List<double>>((row) =>
            (row as List).map((e) => (e as num).toDouble()).toList())
        .toList();
  }

  // Select best person by class 0 score
  const int personClass = 0;
  double bestScore = -1.0;
  _Det? best;
  for (final r in arr) {
    if (r.length < 84) continue;
    final cx = r[0];
    final cy = r[1];
    final w = r[2];
    final h = r[3];
    final clsLogit = r[4 + personClass];
    final score = 1.0 / (1.0 + math.exp(-clsLogit)); // sigmoid
    if (score < 0.25) continue;

    // xywh (letterbox space)
    double x1 = cx - w / 2.0;
    double y1 = cy - h / 2.0;
    double x2 = cx + w / 2.0;
    double y2 = cy + h / 2.0;

    // Map from letterbox -> original image
    double ox1 = (x1 - _lbDw) / _lbR;
    double oy1 = (y1 - _lbDh) / _lbR;
    double ox2 = (x2 - _lbDw) / _lbR;
    double oy2 = (y2 - _lbDh) / _lbR;

    // Clip
    ox1 = ox1.clamp(0.0, _origW.toDouble());
    oy1 = oy1.clamp(0.0, _origH.toDouble());
    ox2 = ox2.clamp(0.0, _origW.toDouble());
    oy2 = oy2.clamp(0.0, _origH.toDouble());
    if (ox2 <= ox1 || oy2 <= oy1) continue;

    if (score > bestScore) {
      bestScore = score;
      best = _Det(ox1, oy1, ox2, oy2, score);
    }
  }

OrtValue _prepRtm(img.Image patch) {
  final h = patch.height;
  final w = patch.width;
  final buf = Float32List(3 * h * w);
  int i = 0;
  for (int c = 0; c < 3; c++) {
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final p = patch.getPixel(x, y);
        final r = img.getRed(p) / 255.0;
        final g = img.getGreen(p) / 255.0;
        final b = img.getBlue(p) / 255.0;
        final v = c == 0 ? r : (c == 1 ? g : b);
        buf[i++] = v;
      }
    }
  }
  return OrtValueTensor.createTensorWithDataList(buf, [1, 3, h, w]);
}

  List<List<double>> _decodeAuto(List<OrtValue?> outs, _CropMeta meta) {
    if (outs.isEmpty || outs[0] == null) {
      return List.generate(17, (_) => [0.0, 0.0, 0.0]);
    }

    final v0 = outs[0]!.value;
    final s0 = _shapeOf(v0);

    if (s0.length == 4) {
      return _decodeHeatmap(outs[0]!, meta);
    }

    if (outs.length >= 2) {
      final s1 = _shapeOf(outs[1]!.value);
      final looksSimCC =
          s0.length == 3 && s1.length == 3 && s0[0] == 1 && s1[0] == 1;
      if (looksSimCC) {
        final xFirst = s0.last <= s1.last;
        return _decodeSimCC(xFirst ? outs : [outs[1], outs[0]], meta);
      }
    }

    return _decodeHeatmap(outs[0]!, meta);
  }

  List<List<double>> _decodeSimCC(List<OrtValue?> outs, _CropMeta meta) {
    final simccX = _toList3D(outs[0]!.value);
    final simccY = _toList3D(outs[1]!.value);
    const split = 2.0;

    final List<List<double>> kpts =
        List.generate(17, (_) => [0.0, 0.0, 0.0]);
    for (int j = 0; j < 17; j++) {
      final rowX = simccX[0][j];
      final rowY = simccY[0][j];
      int ix = 0, iy = 0;
      double vx = -1e9, vy = -1e9;
      for (int t = 0; t < rowX.length; t++) {
        if (rowX[t] > vx) {
          vx = rowX[t];
          ix = t;
        }
      }
      for (int t = 0; t < rowY.length; t++) {
        if (rowY[t] > vy) {
          vy = rowY[t];
          iy = t;
        }
      }
      final px = ix / split;
      final py = iy / split;
      final x = meta.roiX + (px / meta.outW) * meta.roiW;
      final y = meta.roiY + (py / meta.outH) * meta.roiH;
      final conf = math.min(1.0, math.max(0.0, (vx + vy) / 2.0));
      kpts[j][0] = x;
      kpts[j][1] = y;
      kpts[j][2] = conf;
    }
    return kpts;
  }

  List<List<double>> _decodeHeatmap(OrtValue heat, _CropMeta meta) {
    final n1 = heat.value as List;
    if (n1.isEmpty) return List.generate(17, (_) => [0.0, 0.0, 0.0]);
    final nk = n1[0] as List;
    final K = nk.length;
    final H = (nk[0] as List).length;
    final W = ((nk[0] as List)[0] as List).length;

    final List<List<double>> kpts =
        List.generate(17, (_) => [0.0, 0.0, 0.0]);

    for (int j = 0; j < math.min(K, 17); j++) {
      final plane = nk[j] as List;
      double best = -1e9;
      int by = 0, bx = 0;
      for (int y = 0; y < H; y++) {
        final row = plane[y] as List;
        for (int x = 0; x < W; x++) {
          final v = (row[x] as num).toDouble();
          if (v > best) {
            best = v;
            by = y;
            bx = x;
          }
        }
      }
      final px = (bx + 0.5) * (meta.outW / W);
      final py = (by + 0.5) * (meta.outH / H);
      final x = meta.roiX + (px / meta.outW) * meta.roiW;
      final y = meta.roiY + (py / meta.outH) * meta.roiH;
      kpts[j][0] = x;
      kpts[j][1] = y;
      kpts[j][2] = 1.0;
    }
    return kpts;
  }

  List<int> _shapeOf(dynamic v) {
    final dims = <int>[];
    dynamic a = v;
    while (a is List) {
      dims.add(a.length);
      a = a.isNotEmpty ? a[0] : null;
    }
    return dims;
  }

  List<List<double>> _remapToH36M(List<List<double>> rtm) {
    const nose = 0;
    const leftShoulder = 5;
    const rightShoulder = 6;
    const leftElbow = 7;
    const rightElbow = 8;
    const leftWrist = 9;
    const rightWrist = 10;
    const leftHip = 11;
    const rightHip = 12;
    const leftKnee = 13;
    const rightKnee = 14;
    const leftAnkle = 15;
    const rightAnkle = 16;

    List<double> avg(List<List<double>> pts) {
      if (pts.isEmpty) return [0.0, 0.0, 0.0];
      double x = 0, y = 0, c = 0;
      for (final p in pts) {
        x += p[0];
        y += p[1];
        c += p.length > 2 ? p[2] : 0.0;
      }
      final n = pts.length.toDouble();
      return [x / n, y / n, c / n];
    }

    final pelvis = avg([rtm[leftHip], rtm[rightHip]]);
    final thorax = avg([rtm[leftShoulder], rtm[rightShoulder]]);
    final neck = avg([thorax, rtm[nose]]);
    final head = avg([rtm[nose], avg([rtm[leftShoulder], rtm[rightShoulder]])]);

    final h36m = List.generate(17, (_) => [0.0, 0.0, 0.0]);
    h36m[0] = pelvis;
    h36m[1] = List<double>.from(rtm[rightHip]);
    h36m[2] = List<double>.from(rtm[rightKnee]);
    h36m[3] = List<double>.from(rtm[rightAnkle]);
    h36m[4] = List<double>.from(rtm[leftHip]);
    h36m[5] = List<double>.from(rtm[leftKnee]);
    h36m[6] = List<double>.from(rtm[leftAnkle]);
    h36m[7] = avg([pelvis, thorax]);
    h36m[8] = thorax;
    h36m[9] = neck;
    h36m[10] = head;
    h36m[11] = List<double>.from(rtm[leftShoulder]);
    h36m[12] = List<double>.from(rtm[leftElbow]);
    h36m[13] = List<double>.from(rtm[leftWrist]);
    h36m[14] = List<double>.from(rtm[rightShoulder]);
    h36m[15] = List<double>.from(rtm[rightElbow]);
    h36m[16] = List<double>.from(rtm[rightWrist]);
    return h36m;
  }

  _WindowPayload _prepareWindow(List<Pose2DFrame> frames, int start, int T) {
    final list = <Pose2DFrame>[];
    final indices = <int>[];
    for (int offset = 0; offset < T; offset++) {
      final idx = start + offset;
      final mapped = _reflectIndex(idx, frames.length);
      indices.add(mapped);
      list.add(frames[mapped]);
    }
    return _WindowPayload(list, indices);
  }

  int _reflectIndex(int idx, int length) {
    if (length <= 1) return 0;
    if (idx < length) return idx;
    final period = 2 * length - 2;
    var pos = idx % period;
    if (pos < length) return pos;
    return period - pos;
  }

  List<int> _computeWindowStarts(int totalFrames, int window, int stride) {
    if (totalFrames <= window) {
      return [0];
    }
    final starts = <int>[];
    int start = 0;
    while (start + window <= totalFrames) {
      starts.add(start);
      start += stride;
    }
    final lastStart = math.max(0, totalFrames - window);
    if (starts.isEmpty || starts.last != lastStart) {
      starts.add(lastStart);
    }
    return starts;
  }

  OrtValue _toMotionBertInput(List<Pose2DFrame> frames) {
    final T = frames.length;
    final buf = Float32List(1 * T * 17 * 3);
    int i = 0;
    for (final frame in frames) {
      final cx = frame.imgW / 2.0;
      final cy = frame.imgH / 2.0;
      final scale = math.max(1e-6, math.min(frame.imgW, frame.imgH) / 2.0);
      for (final kp in frame.keypoints) {
        buf[i++] = ((kp.x - cx) / scale).toDouble();
        buf[i++] = ((kp.y - cy) / scale).toDouble();
        buf[i++] = (kp.c ?? 1.0).toDouble();
      }
    }
    return OrtValueTensor.createTensorWithDataList(buf, [1, T, 17, 3]);
  }

  List<List<List<double>>> _toList3D(dynamic v) =>
      (v as List)
          .map<List<List<double>>>((a) =>
              (a as List)
                  .map<List<double>>((b) =>
                      (b as List).map<double>((c) => (c as num).toDouble()).toList())
                  .toList())
          .toList();

  List<List<List<List<double>>>> _toList4D(dynamic v, {required List<int> dims}) {
    return (v as List)
        .map<List<List<List<double>>>>((a) =>
            (a as List)
                .map<List<List<double>>>((b) =>
                    (b as List)
                        .map<List<double>>((c) =>
                            (c as List)
                                .map<double>((d) => (d as num).toDouble())
                                .toList())
                        .toList())
                .toList())
        .toList();
  }
}

class _Det {
  _Det(this.x1, this.y1, this.x2, this.y2, this.score);
  final double x1, y1, x2, y2, score;
}

class _CropMeta {
  _CropMeta({
    required this.srcW,
    required this.srcH,
    required this.roiX,
    required this.roiY,
    required this.roiW,
    required this.roiH,
    required this.outW,
    required this.outH,
  });
  final int srcW, srcH, outW, outH;
  final double roiX, roiY, roiW, roiH;
}

class _CropResult {
  _CropResult(this.image, this.meta);
  final img.Image image;
  final _CropMeta meta;
}

class _WindowPayload {
  _WindowPayload(this.frames, this.indices);
  final List<Pose2DFrame> frames;
  final List<int> indices;
}
