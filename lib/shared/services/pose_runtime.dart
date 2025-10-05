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

const bool _enablePoseRuntimeLogs = false;

void _poseRuntimeLog(String message) {
  if (!_enablePoseRuntimeLogs) return;
  debugPrint(message);
}

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
  double _personScoreThr = 0.25;
  double _iouThr = 0.45;
  double _cropScale = 1.25;

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
          _poseRuntimeLog('[PosePipeline] Could not load ${cfg.assetPath}: $e');
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
    double personScore = 0.25,
    double iouThreshold = 0.45,
    double cropScale = 1.25,
  }) async {
    await _ensure2DModelsLoaded();

    _personScoreThr = personScore;
    _iouThr = iouThreshold;
    _cropScale = cropScale;

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
        final crop = _cropForRtm(image, det, outH: 256, outW: 192, scale: _cropScale);
        final rtmIn = _prepRtm(crop.image);
        final rtmOuts = await _run(_rtm!, {'input': rtmIn});
        final raw = _decodeAuto(rtmOuts, crop);
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
    _origW = im.width;
    _origH = im.height;

    const tw = 640, th = 640;
    final r = math.min(th / _origH, tw / _origW);
    final newW = (_origW * r).round();
    final newH = (_origH * r).round();
    final resized = img.copyResize(
      im,
      width: newW,
      height: newH,
      interpolation: img.Interpolation.cubic,
    );
    final canvas = img.Image(width: tw, height: th);
    img.fill(canvas, color: img.ColorRgb8(114, 114, 114));
    final left = ((tw - newW) / 2).floor();
    final top = ((th - newH) / 2).floor();
    img.compositeImage(
      canvas,
      resized,
      dstX: left,
      dstY: top,
      dstW: newW,
      dstH: newH,
      blend: img.BlendMode.direct,
    );

    _lbR = r;
    _lbDw = left.toDouble();
    _lbDh = top.toDouble();
    _lbW = tw;
    _lbH = th;

    final plane = canvas.width * canvas.height;
    final buf = Float32List(3 * plane);
    for (int y = 0; y < canvas.height; y++) {
      for (int x = 0; x < canvas.width; x++) {
        final pixel = canvas.getPixel(x, y);
        final idx = y * canvas.width + x;
        buf[idx] = getRed(pixel) / 255.0;
        buf[plane + idx] = getGreen(pixel) / 255.0;
        buf[(plane * 2) + idx] = getBlue(pixel) / 255.0;
      }
    }
    return OrtValueTensor.createTensorWithDataList(buf, [1, 3, canvas.height, canvas.width]);
  }

  _Det? _pickBestPerson(OrtValue out) {
    final raw = out.value;
    final shape = _shapeOf(raw);
    if (shape.length != 3 || shape[0] != 1) {
      return null;
    }

    List<List<double>> rows = const [];
    if (shape[1] == 84) {
      final channels = (raw as List)[0] as List;
      final count = shape[2];
      rows = List.generate(count, (i) {
        final row = List<double>.filled(84, 0.0);
        for (int c = 0; c < 84; c++) {
          row[c] = ((channels[c] as List)[i] as num).toDouble();
        }
        return row;
      });
    } else if (shape[2] == 84) {
      rows = ((raw as List)[0] as List)
          .map<List<double>>(
              (row) => (row as List).map((e) => (e as num).toDouble()).toList())
          .toList();
    } else {
      return null;
    }

    if (rows.isEmpty) return null;

    final boxes = <List<double>>[];
    final scores = <double>[];
    for (final row in rows) {
      if (row.length < 84) continue;
      final score = 1.0 / (1.0 + math.exp(-row[4]));
      if (score < _personScoreThr) continue;

      final cx = row[0];
      final cy = row[1];
      final w = row[2];
      final h = row[3];

      final x1 = cx - w / 2.0;
      final y1 = cy - h / 2.0;
      final x2 = cx + w / 2.0;
      final y2 = cy + h / 2.0;

      final ox1 = (x1 - _lbDw) / _lbR;
      final oy1 = (y1 - _lbDh) / _lbR;
      final ox2 = (x2 - _lbDw) / _lbR;
      final oy2 = (y2 - _lbDh) / _lbR;

      final clamped = <double>[
        ox1.clamp(0.0, _origW.toDouble()),
        oy1.clamp(0.0, _origH.toDouble()),
        ox2.clamp(0.0, _origW.toDouble()),
        oy2.clamp(0.0, _origH.toDouble()),
      ];
      if (clamped[2] <= clamped[0] || clamped[3] <= clamped[1]) {
        continue;
      }
      boxes.add(clamped);
      scores.add(score);
    }

    if (boxes.isEmpty) return null;

    final keep = _nms(boxes, scores, _iouThr, 50);
    if (keep.isEmpty) return null;
    final bestIdx = keep.first;
    final b = boxes[bestIdx];
    return _Det(b[0], b[1], b[2], b[3], scores[bestIdx]);
  }

  List<int> _shapeOf(Object? v) {
    final dims = <int>[];
    Object? current = v;
    while (current is List && current.isNotEmpty) {
      dims.add(current.length);
      current = current.first;
    }
    if (current is List && current.isEmpty) {
      dims.add(0);
    }
    return dims;
  }

  _CropResult _cropForRtm(
    img.Image im,
    _Det det, {
    int outH = 256,
    int outW = 192,
    double scale = 1.25,
  }) {
    final cx = (det.x1 + det.x2) / 2.0;
    final cy = (det.y1 + det.y2) / 2.0;
    final boxW = math.max(1.0, det.x2 - det.x1);
    final boxH = math.max(1.0, det.y2 - det.y1);

    final desiredAspect = outW / outH;
    double targetW = boxW * scale;
    double targetH = boxH * scale;
    final currentAspect = targetW / targetH;
    if (currentAspect > desiredAspect) {
      targetH = targetW / desiredAspect;
    } else {
      targetW = targetH * desiredAspect;
    }

    double x1 = cx - targetW / 2.0;
    double y1 = cy - targetH / 2.0;
    double x2 = cx + targetW / 2.0;
    double y2 = cy + targetH / 2.0;

    final maxX = im.width.toDouble();
    final maxY = im.height.toDouble();

    if (x1 < 0) {
      x2 -= x1;
      x1 = 0;
    }
    if (y1 < 0) {
      y2 -= y1;
      y1 = 0;
    }
    if (x2 > maxX) {
      final diff = x2 - maxX;
      x1 = math.max(0.0, x1 - diff);
      x2 = maxX;
    }
    if (y2 > maxY) {
      final diff = y2 - maxY;
      y1 = math.max(0.0, y1 - diff);
      y2 = maxY;
    }

    if (x2 <= x1 || y2 <= y1) {
      final resized = img.copyResize(
        im,
        width: outW,
        height: outH,
        interpolation: img.Interpolation.cubic,
      );
      return _CropResult(
        image: resized,
        x: 0.0,
        y: 0.0,
        w: im.width.toDouble(),
        h: im.height.toDouble(),
        outW: outW,
        outH: outH,
      );
    }

    final cropX = x1.floor();
    final cropY = y1.floor();
    int desiredW = (x2 - x1).round();
    int desiredH = (y2 - y1).round();
    if (desiredW < 1) desiredW = 1;
    if (desiredH < 1) desiredH = 1;
    int remainingW = im.width - cropX;
    int remainingH = im.height - cropY;
    if (remainingW < 1) remainingW = 1;
    if (remainingH < 1) remainingH = 1;
    final cropW = remainingW < desiredW ? remainingW : desiredW;
    final cropH = remainingH < desiredH ? remainingH : desiredH;
    final cropped = img.copyCrop(
      im,
      x: cropX,
      y: cropY,
      width: cropW,
      height: cropH,
    );
    final resized = img.copyResize(
      cropped,
      width: outW,
      height: outH,
      interpolation: img.Interpolation.cubic,
    );

    return _CropResult(
      image: resized,
      x: cropX.toDouble(),
      y: cropY.toDouble(),
      w: cropped.width.toDouble(),
      h: cropped.height.toDouble(),
      outW: outW,
      outH: outH,
    );
  }

  OrtValue _prepRtm(img.Image patch) {
    final plane = patch.width * patch.height;
    final buf = Float32List(3 * plane);
    for (int y = 0; y < patch.height; y++) {
      for (int x = 0; x < patch.width; x++) {
        final pixel = patch.getPixel(x, y);
        final idx = y * patch.width + x;
        buf[idx] = getRed(pixel) / 255.0;
        buf[plane + idx] = getGreen(pixel) / 255.0;
        buf[(plane * 2) + idx] = getBlue(pixel) / 255.0;
      }
    }
    return OrtValueTensor.createTensorWithDataList(
      buf,
      [1, 3, patch.height, patch.width],
    );
  }

  List<List<double>> _decodeAuto(List<OrtValue?> outs, _CropResult crop) {
    if (outs.isEmpty || outs[0] == null) {
      return List.generate(17, (_) => [0.0, 0.0, 0.0]);
    }

    final primary = outs[0]!;
    final shape0 = _shapeOf(primary.value);
    if (shape0.length == 4) {
      return _decodeHeatmap(primary, crop);
    }

    if (outs.length >= 2) {
      final secondary = outs[1]!;
      final shape1 = _shapeOf(secondary.value);
      final looksSimcc =
          shape0.length == 3 && shape1.length == 3 && shape0[0] == 1 && shape1[0] == 1;
      if (looksSimcc) {
        final firstIsX = shape0[2] <= shape1[2];
        final ordered = firstIsX ? outs : [outs[1], outs[0]];
        return _decodeSimCC(ordered, crop);
      }
    }

    return _decodeHeatmap(primary, crop);
  }

  List<List<double>> _decodeSimCC(List<OrtValue?> outs, _CropResult crop) {
    final xVal = outs[0]!.value as List;
    final yVal = outs[1]!.value as List;
    if (xVal.isEmpty || yVal.isEmpty) {
      return List.generate(17, (_) => [0.0, 0.0, 0.0]);
    }
    final xData = xVal[0] as List;
    final yData = yVal[0] as List;
    final joints = math.min(17, xData.length);
    final Wx = (xData.first as List).length;
    final Hy = (yData.first as List).length;
    final flatX = Float32List(joints * Wx);
    final flatY = Float32List(joints * Hy);
    for (int j = 0; j < joints; j++) {
      final rowX = xData[j] as List;
      final rowY = yData[j] as List;
      for (int i = 0; i < Wx; i++) {
        flatX[j * Wx + i] = (rowX[i] as num).toDouble();
      }
      for (int i = 0; i < Hy; i++) {
        flatY[j * Hy + i] = (rowY[i] as num).toDouble();
      }
    }

    final decoded = _simccDecode(flatX, flatY, Wx: Wx, Hy: Hy, split: 2.0);
    final result = List.generate(17, (_) => [0.0, 0.0, 0.0]);
    for (int j = 0; j < decoded.length && j < 17; j++) {
      final joint = decoded[j];
      final px = joint[0];
      final py = joint[1];
      final conf = joint[2];
      result[j][0] = crop.x + (px / crop.outW) * crop.w;
      result[j][1] = crop.y + (py / crop.outH) * crop.h;
      result[j][2] = conf;
    }
    return result;
  }

  List<List<double>> _decodeHeatmap(OrtValue heat, _CropResult crop) {
    final arr = heat.value as List;
    if (arr.isEmpty) {
      return List.generate(17, (_) => [0.0, 0.0, 0.0]);
    }
    final planes = arr[0] as List;
    final joints = math.min(17, planes.length);
    final result = List.generate(17, (_) => [0.0, 0.0, 0.0]);
    for (int j = 0; j < joints; j++) {
      final plane = planes[j] as List;
      final H = plane.length;
      final W = (plane[0] as List).length;
      double bestVal = -double.infinity;
      int bestX = 0;
      int bestY = 0;
      for (int y = 0; y < H; y++) {
        final row = plane[y] as List;
        for (int x = 0; x < W; x++) {
          final v = (row[x] as num).toDouble();
          if (v > bestVal) {
            bestVal = v;
            bestX = x;
            bestY = y;
          }
        }
      }
      final px = (bestX + 0.5) * (crop.outW / W);
      final py = (bestY + 0.5) * (crop.outH / H);
      result[j][0] = crop.x + (px / crop.outW) * crop.w;
      result[j][1] = crop.y + (py / crop.outH) * crop.h;
      result[j][2] = 1.0;
    }
    return result;
  }

  List<List<double>> _simccDecode(
    Float32List simccX,
    Float32List simccY, {
    int Wx = 384,
    int Hy = 512,
    double split = 2.0,
  }) {
    final joints = math.min(simccX.length ~/ Wx, simccY.length ~/ Hy);
    final result = List.generate(joints, (_) => [0.0, 0.0, 0.0]);
    for (int j = 0; j < joints; j++) {
      final baseX = j * Wx;
      final baseY = j * Hy;

      double maxX = -double.infinity;
      for (int i = 0; i < Wx; i++) {
        maxX = math.max(maxX, simccX[baseX + i]);
      }
      double sumX = 0.0;
      double bestProbX = 0.0;
      int argX = 0;
      for (int i = 0; i < Wx; i++) {
        final expVal = math.exp(simccX[baseX + i] - maxX);
        sumX += expVal;
        if (expVal > bestProbX) {
          bestProbX = expVal;
          argX = i;
        }
      }
      final probX = sumX == 0 ? 0.0 : bestProbX / sumX;

      double maxY = -double.infinity;
      for (int i = 0; i < Hy; i++) {
        maxY = math.max(maxY, simccY[baseY + i]);
      }
      double sumY = 0.0;
      double bestProbY = 0.0;
      int argY = 0;
      for (int i = 0; i < Hy; i++) {
        final expVal = math.exp(simccY[baseY + i] - maxY);
        sumY += expVal;
        if (expVal > bestProbY) {
          bestProbY = expVal;
          argY = i;
        }
      }
      final probY = sumY == 0 ? 0.0 : bestProbY / sumY;

      final x = argX / split;
      final y = argY / split;
      final conf = math.sqrt(probX * probY);
      result[j][0] = x;
      result[j][1] = y;
      result[j][2] = conf.isNaN ? 0.0 : conf.clamp(0.0, 1.0);
    }
    return result;
  }

  List<int> _nms(
    List<List<double>> boxes,
    List<double> scores,
    double iouThr,
    int maxDet,
  ) {
    final order = List<int>.generate(scores.length, (i) => i)
      ..sort((a, b) => scores[b].compareTo(scores[a]));
    final keep = <int>[];
    while (order.isNotEmpty && keep.length < maxDet) {
      final i = order.removeAt(0);
      keep.add(i);
      order.removeWhere((j) => _iou(boxes[i], boxes[j]) > iouThr);
    }
    return keep;
  }

  double _iou(List<double> a, List<double> b) {
    final x1 = math.max(a[0], b[0]);
    final y1 = math.max(a[1], b[1]);
    final x2 = math.min(a[2], b[2]);
    final y2 = math.min(a[3], b[3]);
    final interW = math.max(0.0, x2 - x1);
    final interH = math.max(0.0, y2 - y1);
    final inter = interW * interH;
    if (inter <= 0) return 0.0;
    final areaA = (a[2] - a[0]) * (a[3] - a[1]);
    final areaB = (b[2] - b[0]) * (b[3] - b[1]);
    final union = areaA + areaB - inter;
    if (union <= 0) return 0.0;
    return inter / union;
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

class _CropResult {
  const _CropResult({
    required this.image,
    required this.x,
    required this.y,
    required this.w,
    required this.h,
    required this.outW,
    required this.outH,
  });

  final img.Image image;
  final double x;
  final double y;
  final double w;
  final double h;
  final int outW;
  final int outH;
}

class _WindowPayload {
  _WindowPayload(this.frames, this.indices);
  final List<Pose2DFrame> frames;
  final List<int> indices;
}