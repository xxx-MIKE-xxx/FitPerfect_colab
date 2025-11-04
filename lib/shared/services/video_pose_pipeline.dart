import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';
import 'package:path/path.dart' as p;

import 'ort_session.dart';
import 'pose_runtime.dart';
import 'storage_layout.dart';
import 'video_sampler.dart' as sampler;

typedef FrameBatch = sampler.VideoFrameBatch;

class VideoPipelineSummary {
  VideoPipelineSummary({
    required this.sessionId,
    required this.poseResult,
    required this.frames2d,
    required this.windows3d,
    required this.videoFile,
    required this.out2d,
    required this.out3d,
    required this.metaFile,
  });

  final String sessionId;
  final PosePipelineResult poseResult;
  final int frames2d;
  final int windows3d;
  final File videoFile;
  final File out2d;
  final File out3d;
  final File metaFile;
}

class VideoPosePipeline {
  VideoPosePipeline({
    this.rtmModel = 'assets/models/rtmpose-m_256x192.onnx',
    this.sampleInterval = const Duration(milliseconds: 125),
    this.maxDetectionAge = const Duration(milliseconds: 500),
  });

  final String rtmModel;
  final Duration sampleInterval;
  final Duration maxDetectionAge;

  OrtSession? _rtm;

  Future<void> _ensureRtmLoaded() async {
    _rtm ??= await OrtManager.fromAsset(rtmModel);
  }

  Future<VideoPipelineSummary> run({
    required File video,
    required String sessionId,
  }) async {
    await _ensureRtmLoaded();

    final Directory sessionDir = await StorageLayout.sessionDir(sessionId);
    final File sessionVideo = File(p.join(sessionDir.path, 'video.mp4'));
    await video.copy(sessionVideo.path);

    final File yoloFile = await StorageLayout.yoloDetectionsFile(sessionId);
    if (!await yoloFile.exists()) {
      throw StateError('Missing YOLO detections at ${yoloFile.path}');
    }

    final List<_YoloRecord> detections = await _readYoloDetections(yoloFile);
    if (detections.isEmpty) {
      throw StateError('No person detections found in ${yoloFile.path}');
    }

    final int dtMs = sampleInterval.inMilliseconds;
    final double sampleFps = dtMs > 0 ? 1000.0 / dtMs : 8.0;

    final sampler.VideoFrameBatch batch =
        await sampler.VideoSampler.extractFramesAtFps(sessionVideo, sampleFps);

    final File out2d = await StorageLayout.out2dFile(sessionId);
    final IOSink out2dSink = out2d.openWrite(mode: FileMode.write);

    final List<Pose2DFrame> frames = [];
    int frameWidth = 0;
    int frameHeight = 0;

    int detectionCursor = 0;
    _YoloRecord? currentDetection;

    for (int i = 0; i < batch.files.length; i++) {
      final File frameFile = batch.files[i];
      final Uint8List bytes = await frameFile.readAsBytes();
      final img.Image? image = img.decodeImage(bytes);
      if (image == null) {
        continue;
      }

      frameWidth = image.width;
      frameHeight = image.height;

      final int tsMs = i * dtMs;
      while (detectionCursor < detections.length &&
          detections[detectionCursor].tsMs <= tsMs) {
        currentDetection = detections[detectionCursor];
        detectionCursor++;
      }

      final _YoloRecord? det = currentDetection;
      if (det == null) {
        continue;
      }
      if (tsMs - det.tsMs > maxDetectionAge.inMilliseconds) {
        continue;
      }

      final _CropResult crop = _cropImage(image, det);
      if (crop.image.width == 0 || crop.image.height == 0) {
        continue;
      }

      final _LetterboxResult lb = _letterbox(crop.image, outW: 192, outH: 256);
      final OrtValue input = _prepRtm(lb.image);
      final List<OrtValue?> outs = await _run(_rtm!, {'input': input});
      if (outs.isEmpty || outs.first == null) {
        continue;
      }
      final List<List<double>> kpts = _decodeAuto(outs, lb.meta);
      for (final OrtValue? v in outs) {
        v?.release();
      }

      final List<List<double>> kptsRaw = _restoreToRaw(kpts, lb.meta, crop);
      final Map<String, dynamic> record = {
        't': double.parse((tsMs / 1000.0).toStringAsFixed(3)),
        'ts_ms': tsMs,
        'frame_idx': det.frameIdx,
        'frame_size': [det.frameHeight ?? frameHeight, det.frameWidth ?? frameWidth],
        'bbox_norm': det.bboxNorm ?? _normalize(det.bboxRaw, det.frameWidth ?? frameWidth, det.frameHeight ?? frameHeight),
        'lb_params': lb.meta.toJson(),
        'kpt_coco': kptsRaw
            .map((kp) => kp.map((v) => double.parse(v.toStringAsFixed(5))).toList())
            .toList(),
      };
      out2dSink.writeln(jsonEncode(record));

      final List<List<double>> h36m = _cocoToH36M(kptsRaw);
      frames.add(
        Pose2DFrame(
          frameIndex: det.frameIdx,
          t: tsMs / 1000.0,
          keypoints: List.generate(17, (j) {
            final kp = h36m[j];
            return PoseKeypoint2D(id: j, x: kp[0], y: kp[1], c: kp[2]);
          }),
          imgW: det.frameWidth ?? frameWidth,
          imgH: det.frameHeight ?? frameHeight,
        ),
      );
    }

    await out2dSink.flush();
    await out2dSink.close();
    await batch.cleanup();

    final Pose2DSequence seq = Pose2DSequence(
      frames: frames,
      fps: sampleFps,
      imageWidth: frameWidth,
      imageHeight: frameHeight,
    );

    final PosePipeline posePipeline = PosePipeline();
    final Pose3DResult pose3d = await posePipeline.estimate3D(
      seq,
      pelvisCentered: true,
    );

    final PosePipelineResult result =
        PosePipelineResult(pose2d: seq, pose3d: pose3d);

    final File out3d = await _writeOut3d(sessionId, pose3d);
    final File metaFile = await _writeMeta(
      sessionId: sessionId,
      sampleFps: sampleFps,
      frameWidth: frameWidth,
      frameHeight: frameHeight,
      frames2d: frames.length,
      windows3d: pose3d.windows.length,
    );

    return VideoPipelineSummary(
      sessionId: sessionId,
      poseResult: result,
      frames2d: frames.length,
      windows3d: pose3d.windows.length,
      videoFile: sessionVideo,
      out2d: out2d,
      out3d: out3d,
      metaFile: metaFile,
    );
  }

  Future<List<OrtValue?>> _run(
    OrtSession session,
    Map<String, OrtValue> inputs,
  ) async {
    final opts = OrtRunOptions();
    final outs = await session.runAsync(opts, inputs);
    opts.release();
    for (final entry in inputs.values) {
      entry.release();
    }
    return outs ?? const [];
  }

  Future<List<_YoloRecord>> _readYoloDetections(File file) async {
    final List<_YoloRecord> records = [];
    await for (final line in file.openRead()
        .transform(utf8.decoder)
        .transform(const LineSplitter())) {
      if (line.trim().isEmpty) continue;
      final dynamic obj = json.decode(line);
      if (obj is Map<String, dynamic>) {
        final rec = _YoloRecord.fromJson(obj);
        if (rec != null) {
          records.add(rec);
        }
      }
    }
    records.sort((a, b) => a.tsMs.compareTo(b.tsMs));
    return records;
  }

  _CropResult _cropImage(img.Image image, _YoloRecord det) {
    final double x1 = det.bboxRaw[0];
    final double y1 = det.bboxRaw[1];
    final double x2 = det.bboxRaw[2];
    final double y2 = det.bboxRaw[3];

    final int ix1 = x1.floor().clamp(0, image.width - 1);
    final int iy1 = y1.floor().clamp(0, image.height - 1);
    final int ix2 = x2.ceil().clamp(ix1 + 1, image.width);
    final int iy2 = y2.ceil().clamp(iy1 + 1, image.height);

    final img.Image crop = img.copyCrop(
      image,
      x: ix1,
      y: iy1,
      width: (ix2 - ix1).clamp(1, image.width),
      height: (iy2 - iy1).clamp(1, image.height),
    );
    return _CropResult(
      image: crop,
      meta: _CropMeta(
        x1: ix1.toDouble(),
        y1: iy1.toDouble(),
        width: crop.width,
        height: crop.height,
      ),
    );
  }

  _LetterboxResult _letterbox(
    img.Image src, {
    required int outW,
    required int outH,
  }) {
    final double r = math.min(outW / src.width, outH / src.height);
    final int newW = (src.width * r).round().clamp(1, outW);
    final int newH = (src.height * r).round().clamp(1, outH);

    final img.Image resized = img.copyResize(
      src,
      width: newW,
      height: newH,
      interpolation: img.Interpolation.cubic,
    );

    final img.Image canvas = img.Image(width: outW, height: outH);
    img.fill(canvas, color: img.ColorRgb8(114, 114, 114));

    final double dw = (outW - newW) / 2.0;
    final double dh = (outH - newH) / 2.0;

    img.compositeImage(canvas, resized, dstX: dw.floor(), dstY: dh.floor());

    return _LetterboxResult(
      image: canvas,
      meta: _LetterboxMeta(
        r: r,
        dw: dw,
        dh: dh,
        inW: src.width,
        inH: src.height,
        outW: outW,
        outH: outH,
      ),
    );
  }

  OrtValue _prepRtm(img.Image patch) {
    final int w = patch.width;
    final int h = patch.height;
    final int plane = w * h;
    final Float32List buf = Float32List(plane * 3);
    const List<double> mean = [0.485, 0.456, 0.406];
    const List<double> std = [0.229, 0.224, 0.225];

    final Uint8List rgb = patch.getBytes(order: img.ChannelOrder.rgb);
    for (int i = 0, p = 0; p < plane; p++) {
      final double r = rgb[i++] / 255.0;
      final double g = rgb[i++] / 255.0;
      final double b = rgb[i++] / 255.0;
      buf[p] = (r - mean[0]) / std[0];
      buf[plane + p] = (g - mean[1]) / std[1];
      buf[(plane * 2) + p] = (b - mean[2]) / std[2];
    }
    return OrtValueTensor.createTensorWithDataList(buf, [1, 3, h, w]);
  }

  List<List<double>> _decodeAuto(List<OrtValue?> outs, _LetterboxMeta meta) {
    if (outs.isEmpty || outs.first == null) {
      return List.generate(17, (_) => [0.0, 0.0, 0.0]);
    }
    final OrtValue out = outs.first!;
    final List<int> shape = _shapeOf(out.value);
    if (shape.length == 4 && shape[1] == 17 && shape.last == 2) {
      // SimCC: [1,17,2,256]
      return _decodeSimcc(out.value, shape, meta);
    }
    if (shape.length == 4 && shape[1] == 17) {
      // Heatmap
      return _decodeHeatmap(out.value, shape, meta);
    }
    return List.generate(17, (_) => [0.0, 0.0, 0.0]);
  }

  List<List<double>> _decodeHeatmap(
    dynamic value,
    List<int> shape,
    _LetterboxMeta meta,
  ) {
    final int H = shape[2];
    final int W = shape[3];
    final List<List<double>> points = List.generate(17, (_) => [0.0, 0.0, 0.0]);
    for (int k = 0; k < 17; k++) {
      double best = -double.infinity;
      int bestX = 0;
      int bestY = 0;
      for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
          final double score = ((value[0][k][y][x]) as num).toDouble();
          if (score > best) {
            best = score;
            bestX = x;
            bestY = y;
          }
        }
      }
      final double px = bestX * (meta.outW / W) + meta.dw;
      final double py = bestY * (meta.outH / H) + meta.dh;
      points[k][0] = px;
      points[k][1] = py;
      points[k][2] = best;
    }
    return points;
  }

  List<List<double>> _decodeSimcc(
    dynamic value,
    List<int> shape,
    _LetterboxMeta meta,
  ) {
    final int bins = shape[3];
    final List<List<double>> points = List.generate(17, (_) => [0.0, 0.0, 0.0]);
    for (int k = 0; k < 17; k++) {
      final List xBins = value[0][k][0] as List;
      final List yBins = value[0][k][1] as List;
      int argMaxX = 0;
      int argMaxY = 0;
      double maxX = -double.infinity;
      double maxY = -double.infinity;
      for (int i = 0; i < bins; i++) {
        final double vx = (xBins[i] as num).toDouble();
        if (vx > maxX) {
          maxX = vx;
          argMaxX = i;
        }
        final double vy = (yBins[i] as num).toDouble();
        if (vy > maxY) {
          maxY = vy;
          argMaxY = i;
        }
      }
      final double px = argMaxX * (meta.outW / bins) + meta.dw;
      final double py = argMaxY * (meta.outH / bins) + meta.dh;
      points[k][0] = px;
      points[k][1] = py;
      points[k][2] = math.min(maxX, maxY);
    }
    return points;
  }

  List<int> _shapeOf(dynamic value) {
    final List<int> dims = [];
    dynamic cur = value;
    while (cur is List && cur.isNotEmpty) {
      dims.add(cur.length);
      cur = cur.first;
    }
    return dims;
  }

  List<List<double>> _restoreToRaw(
    List<List<double>> kpts,
    _LetterboxMeta lb,
    _CropResult crop,
  ) {
    final List<List<double>> result =
        List.generate(kpts.length, (_) => [0.0, 0.0, 0.0]);
    for (int i = 0; i < kpts.length; i++) {
      final double x = (kpts[i][0] - lb.dw) / lb.r;
      final double y = (kpts[i][1] - lb.dh) / lb.r;
      final double rawX = (x + crop.meta.x1).clamp(0.0, double.infinity);
      final double rawY = (y + crop.meta.y1).clamp(0.0, double.infinity);
      result[i][0] = rawX;
      result[i][1] = rawY;
      result[i][2] = kpts[i][2];
    }
    return result;
  }

  List<List<double>> _cocoToH36M(List<List<double>> coco) {
    double px(int i) => coco[i][0];
    double py(int i) => coco[i][1];
    double pc(int i) => coco[i][2];

    final pelvisX = (px(11) + px(12)) / 2.0;
    final pelvisY = (py(11) + py(12)) / 2.0;
    final neckX = (px(5) + px(6)) / 2.0;
    final neckY = (py(5) + py(6)) / 2.0;
    final spineX = (pelvisX + neckX) / 2.0;
    final spineY = (pelvisY + neckY) / 2.0;

    final headPoints = [1, 2, 3, 4]
        .where((i) => pc(i) > 0)
        .map((i) => [px(i), py(i)])
        .toList();
    final headX = headPoints.isEmpty
        ? px(0)
        : headPoints.map((p) => p[0]).reduce((a, b) => a + b) / headPoints.length;
    final headY = headPoints.isEmpty
        ? py(0)
        : headPoints.map((p) => p[1]).reduce((a, b) => a + b) / headPoints.length;

    final List<List<double>> h36m = List.generate(17, (_) => [0.0, 0.0, 0.0]);

    void setJoint(int idx, double x, double y, double c) {
      h36m[idx][0] = x;
      h36m[idx][1] = y;
      h36m[idx][2] = c;
    }

    setJoint(0, pelvisX, pelvisY, (pc(11) + pc(12)) / 2.0);
    setJoint(1, px(12), py(12), pc(12));
    setJoint(2, px(14), py(14), pc(14));
    setJoint(3, px(16), py(16), pc(16));
    setJoint(4, px(11), py(11), pc(11));
    setJoint(5, px(13), py(13), pc(13));
    setJoint(6, px(15), py(15), pc(15));
    setJoint(7, spineX, spineY, (pc(11) + pc(12) + pc(5) + pc(6)) / 4.0);
    setJoint(8, neckX, neckY, (pc(5) + pc(6)) / 2.0);
    setJoint(9, headX, headY, pc(0));
    setJoint(10, px(0), py(0), pc(0));
    setJoint(11, px(5), py(5), pc(5));
    setJoint(12, px(7), py(7), pc(7));
    setJoint(13, px(9), py(9), pc(9));
    setJoint(14, px(6), py(6), pc(6));
    setJoint(15, px(8), py(8), pc(8));
    setJoint(16, px(10), py(10), pc(10));

    return h36m;
  }

  List<double> _normalize(List<double> bbox, int width, int height) {
    return [
      bbox[0] / width,
      bbox[1] / height,
      bbox[2] / width,
      bbox[3] / height,
    ]
        .map((v) => double.parse(v.toStringAsFixed(6)))
        .toList();
  }

  Future<File> _writeOut3d(String sessionId, Pose3DResult pose3d) async {
    final File out3d = await StorageLayout.out3dFile(sessionId);
    final tmp = File('${out3d.path}.tmp');
    final Map<String, dynamic> payload = {
      'T': pose3d.sequence.length,
      'h36m_order': const [
        'Pelvis',
        'RHip',
        'RKnee',
        'RAnkle',
        'LHip',
        'LKnee',
        'LAnkle',
        'Spine1',
        'Neck',
        'Head',
        'Site',
        'LShoulder',
        'LElbow',
        'LWrist',
        'RShoulder',
        'RElbow',
        'RWrist'
      ],
      'coords_3d': pose3d.sequence,
    };
    await tmp.writeAsString(const JsonEncoder.withIndent('  ').convert(payload));
    if (await out3d.exists()) {
      await out3d.delete();
    }
    await tmp.rename(out3d.path);
    return out3d;
  }

  Future<File> _writeMeta({
    required String sessionId,
    required double sampleFps,
    required int frameWidth,
    required int frameHeight,
    required int frames2d,
    required int windows3d,
  }) async {
    final File meta = await StorageLayout.metaFile(sessionId);
    final double durationSec = frames2d <= 1
        ? (frames2d == 1 ? sampleInterval.inMilliseconds / 1000.0 : 0.0)
        : ((frames2d - 1) * sampleInterval.inMilliseconds) / 1000.0;
    final Map<String, dynamic> payload = {
      'session_id': sessionId,
      'video': {
        'fps': double.parse(sampleFps.toStringAsFixed(3)),
        'size': [frameHeight, frameWidth],
        'dt_ms': sampleInterval.inMilliseconds,
        'duration_s': double.parse(durationSec.toStringAsFixed(3)),
      },
      'sampling': {
        'strategy': 'timestamp',
        'dt_ms': sampleInterval.inMilliseconds,
      },
      'models': {
        'yolo': {'name': 'yolov8n-person.onnx', 'input': [640, 640]},
        'rtmpose': {'name': p.basename(rtmModel), 'input': [256, 192]},
        'motionbert': {'name': 'MB_ft_h36m.bin', 'clip_len_max': 243, 'stride': 81},
      },
      'counts': {
        'frames_2d': frames2d,
        'clips_3d': windows3d,
      },
    };
    await meta.writeAsString(const JsonEncoder.withIndent('  ').convert(payload));
    return meta;
  }
}

class _YoloRecord {
  _YoloRecord({
    required this.t,
    required this.tsMs,
    required this.frameIdx,
    required this.bboxRaw,
    required this.score,
    this.frameWidth,
    this.frameHeight,
    this.bboxNorm,
  });

  final double t;
  final int tsMs;
  final int frameIdx;
  final List<double> bboxRaw;
  final double score;
  final int? frameWidth;
  final int? frameHeight;
  final List<double>? bboxNorm;

  static _YoloRecord? fromJson(Map<String, dynamic> json) {
    final List<dynamic>? raw = json['bbox_raw'] as List<dynamic>?;
    if (raw == null || raw.length != 4) return null;
    final List<double> bbox =
        raw.map((e) => (e as num).toDouble()).toList(growable: false);
    final frameSize = json['frame_size'] as List<dynamic>?;
    final width = frameSize != null && frameSize.length >= 2
        ? (frameSize[1] as num).toInt()
        : null;
    final height = frameSize != null && frameSize.length >= 2
        ? (frameSize[0] as num).toInt()
        : null;
    final norm = json['bbox_norm'] as List<dynamic>?;
    return _YoloRecord(
      t: (json['t'] as num?)?.toDouble() ?? 0.0,
      tsMs: (json['ts_ms'] as num?)?.toInt() ?? 0,
      frameIdx: (json['frame_idx'] as num?)?.toInt() ?? 0,
      bboxRaw: bbox,
      score: (json['score'] as num?)?.toDouble() ?? 0.0,
      frameWidth: width,
      frameHeight: height,
      bboxNorm: norm?.map((e) => (e as num).toDouble()).toList(),
    );
  }
}

class _CropMeta {
  _CropMeta({
    required this.x1,
    required this.y1,
    required this.width,
    required this.height,
  });

  final double x1;
  final double y1;
  final int width;
  final int height;
}

class _CropResult {
  _CropResult({required this.image, required this.meta});
  final img.Image image;
  final _CropMeta meta;
}

class _LetterboxMeta {
  _LetterboxMeta({
    required this.r,
    required this.dw,
    required this.dh,
    required this.inW,
    required this.inH,
    required this.outW,
    required this.outH,
  });

  final double r;
  final double dw;
  final double dh;
  final int inW;
  final int inH;
  final int outW;
  final int outH;

  Map<String, dynamic> toJson() => {
        'r': double.parse(r.toStringAsFixed(6)),
        'dw': double.parse(dw.toStringAsFixed(3)),
        'dh': double.parse(dh.toStringAsFixed(3)),
        'in_w': inW,
        'in_h': inH,
        'out': [outH, outW],
      };
}

class _LetterboxResult {
  _LetterboxResult({required this.image, required this.meta});
  final img.Image image;
  final _LetterboxMeta meta;
}
