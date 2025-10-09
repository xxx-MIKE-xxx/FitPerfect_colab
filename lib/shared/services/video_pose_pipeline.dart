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
import 'video_sampler.dart';

class PreflightSummary {
  PreflightSummary({
    required this.medianMs,
    required this.p90Ms,
    required this.fpsChainMax,
    required this.fpsTarget,
    required this.dtMs,
    required this.mode,
  });

  final double medianMs;
  final double p90Ms;
  final double fpsChainMax;
  final int fpsTarget;
  final int dtMs;
  final String mode;

  Map<String, dynamic> toJson() => {
        'median_ms': double.parse(medianMs.toStringAsFixed(3)),
        'p90_ms': double.parse(p90Ms.toStringAsFixed(3)),
        'fps_chain_max': double.parse(fpsChainMax.toStringAsFixed(3)),
        'fps_target': fpsTarget,
        'dt_ms': dtMs,
        'mode': mode,
      };
}

class VideoPipelineSummary {
  VideoPipelineSummary({
    required this.sessionId,
    required this.poseResult,
    required this.preflight,
    required this.frames2d,
    required this.windows3d,
    required this.videoFile,
    required this.out2d,
    required this.out3d,
    required this.metaFile,
  });

  final String sessionId;
  final PosePipelineResult poseResult;
  final PreflightSummary preflight;
  final int frames2d;
  final int windows3d;
  final File videoFile;
  final File out2d;
  final File out3d;
  final File metaFile;
}

class VideoPosePipeline {
  VideoPosePipeline({
    this.yoloModel = 'assets/models/yolov8n.onnx',
    this.rtmModel = 'assets/models/rtmpose-m_256x192.onnx',
    double roiMargin = 1.25,
  }) : _roiMargin = roiMargin;

  final String yoloModel;
  final String rtmModel;
  final double _roiMargin;

  OrtSession? _yolo;
  OrtSession? _rtm;
  final PosePipeline _posePipeline = PosePipeline();

  Future<void> _ensure2DModelsLoaded() async {
    _yolo ??= await OrtManager.fromAsset(yoloModel);
    _rtm ??= await OrtManager.fromAsset(rtmModel);
  }

  Future<VideoPipelineSummary> run({
    required File video,
    required String sessionId,
  }) async {
    await _ensure2DModelsLoaded();

    final Directory sessionDir = await StorageLayout.sessionDir(sessionId);
    final File sessionVideo = File(p.join(sessionDir.path, 'video.mp4'));
    await video.copy(sessionVideo.path);

    final img.Image? referenceFrame = await _loadReferenceFrame(video);
    if (referenceFrame == null) {
      throw StateError('Could not decode reference frame from video');
    }

    final PreflightSummary preflight =
        await _runPreflight(referenceFrame, budget: const Duration(milliseconds: 1500));
    final int fpsTarget = math.max(1, preflight.fpsTarget);
    final int dtMs = preflight.dtMs > 0 ? preflight.dtMs : (1000 / fpsTarget).round();

    final VideoFrameBatch batch =
        await VideoSampler.extractFramesAtFps(video, fpsTarget.toDouble());

    final File out2d = await StorageLayout.out2dFile(sessionId);
    final IOSink out2dSink = out2d.openWrite(mode: FileMode.write);

    final List<Pose2DFrame> frames = [];
    int frameWidth = referenceFrame.width;
    int frameHeight = referenceFrame.height;
    int tsMs = 0;

    for (int i = 0; i < batch.files.length; i++) {
      final File file = batch.files[i];
      final Uint8List bytes = await file.readAsBytes();
      final img.Image? image = img.decodeImage(bytes);
      if (image == null) {
        continue;
      }

      frameWidth = image.width;
      frameHeight = image.height;

      final _ChainData chain = await _runChain(image);
      final List<List<double>> coco = chain.coco;
      final List<List<double>> h36m = chain.h36m;

      final Pose2DRecord record = Pose2DRecord(
        t: tsMs / 1000.0,
        tsMs: tsMs,
        frameIndex: i,
        frameSize: [image.width, image.height],
        bbox: chain.bbox,
        letterbox: chain.letterbox,
        keypoints: coco,
      );
      out2dSink.writeln(jsonEncode(record.toJson()));

      frames.add(
        Pose2DFrame(
          frameIndex: i,
          t: tsMs / 1000.0,
          keypoints: List.generate(17, (j) {
            final kp = h36m[j];
            return PoseKeypoint2D(id: j, x: kp[0], y: kp[1], c: kp[2]);
          }),
          imgW: image.width,
          imgH: image.height,
        ),
      );

      tsMs += dtMs;
    }

    await out2dSink.flush();
    await out2dSink.close();

    await batch.cleanup();

    final File cocoShadow = await StorageLayout.cocoShadowFile(sessionId);
    try {
      await out2d.copy(cocoShadow.path);
    } catch (_) {}

    final Pose2DSequence pose2d = Pose2DSequence(
      frames: frames,
      fps: fpsTarget.toDouble(),
      imageWidth: frameWidth,
      imageHeight: frameHeight,
    );

    final Pose3DResult pose3d =
        await _posePipeline.estimate3D(pose2d, pelvisCentered: true);

    final File out3d = await StorageLayout.out3dFile(sessionId);
    await out3d.writeAsString(jsonEncode(pose3d.sequence));

    final PosePipelineResult poseResult =
        PosePipelineResult(pose2d: pose2d, pose3d: pose3d);

    final File metaFile = await StorageLayout.metaFile(sessionId);
    final MotionBertConfig mbCfg = _posePipeline.motionBertConfig;

    await metaFile.writeAsString(
      const JsonEncoder.withIndent('  ').convert({
        'session_id': sessionId,
        'video': {
          'path': sessionVideo.path,
          'fps_target': fpsTarget,
          'dt_ms': dtMs,
          'size': [frameWidth, frameHeight],
          'duration_s': frames.isEmpty ? 0.0 : double.parse(((frames.length - 1) * dtMs / 1000).toStringAsFixed(3)),
        },
        'preflight': preflight.toJson(),
        'sampling': {
          'strategy': 'timestamp',
          'dt_ms': dtMs,
          'fps': fpsTarget,
        },
        'models': {
          'yolo': {'name': p.basename(yoloModel), 'input': [640, 640]},
          'rtmpose': {'name': p.basename(rtmModel), 'input': [256, 192]},
          'motionbert': {
            'name': p.basename(mbCfg.assetPath),
            'clip_len_max': mbCfg.window,
            'stride': mbCfg.stride,
          },
        },
        'counts': {
          'frames_2d': frames.length,
          'clips_3d': pose3d.windows.length,
        },
      }),
    );

    return VideoPipelineSummary(
      sessionId: sessionId,
      poseResult: poseResult,
      preflight: preflight,
      frames2d: frames.length,
      windows3d: pose3d.windows.length,
      videoFile: sessionVideo,
      out2d: out2d,
      out3d: out3d,
      metaFile: metaFile,
    );
  }

  Future<img.Image?> _loadReferenceFrame(File video) async {
    final VideoFrameBatch batch =
        await VideoSampler.extractFramesAtFps(video, 1.0, maxFrames: 1);
    try {
      if (batch.files.isEmpty) {
        return null;
      }
      final Uint8List bytes = await batch.files.first.readAsBytes();
      return img.decodeImage(bytes);
    } finally {
      await batch.cleanup();
    }
  }

  Future<PreflightSummary> _runPreflight(
    img.Image frame, {
    required Duration budget,
  }) async {
    final List<double> samples = [];

    await _runChain(frame); // warmup

    final Stopwatch timer = Stopwatch()..start();
    while (timer.elapsed < budget) {
      final Stopwatch sw = Stopwatch()..start();
      await _runChain(frame);
      samples.add(sw.elapsedMicroseconds / 1000.0);
    }

    if (samples.isEmpty) {
      samples.add(1.0);
    }

    samples.sort();
    final double median = samples[samples.length ~/ 2];
    final int p90Index = ((samples.length - 1) * 0.9).round();
    final double p90 = samples[p90Index.clamp(0, samples.length - 1)];
    final double fpsChainMax = median > 0 ? 1000.0 / median : 0.0;
    final int fpsTarget = fpsChainMax.isFinite && fpsChainMax > 0
        ? math.max(1, (0.8 * fpsChainMax).floor())
        : 1;
    final int dtMs = fpsTarget > 0 ? (1000 / fpsTarget).round() : 125;
    final String mode = fpsTarget >= 8 ? 'live' : 'post';

    return PreflightSummary(
      medianMs: median,
      p90Ms: p90,
      fpsChainMax: fpsChainMax,
      fpsTarget: fpsTarget,
      dtMs: dtMs,
      mode: mode,
    );
  }

  Future<_ChainData> _runChain(img.Image image) async {
    final _LetterboxResult lb = _letterbox(image);
    final OrtValue yoloInput = _prepYolo(lb.image);
    final List<OrtValue?> yoloOuts = await _run(_yolo!, {'images': yoloInput});

    _Det? det;
    if (yoloOuts.isNotEmpty && yoloOuts.first != null) {
      det = _pickLargestPerson(yoloOuts.first!, lb.meta, image.width, image.height);
    }
    for (final OrtValue? v in yoloOuts) {
      v?.release();
    }

    if (det == null) {
      return _ChainData(
        letterbox: lb.meta,
        bbox: null,
        coco: List.generate(17, (_) => [0.0, 0.0, 0.0]),
        h36m: List.generate(17, (_) => [0.0, 0.0, 0.0]),
      );
    }

    final _Det detLb = _rawToLb(det, lb.meta, clamp: true, margin: _roiMargin);
    final _CropResult crop = _cropForRtm(lb.image, detLb, outH: 256, outW: 192);
    final OrtValue rtmInput = _prepRtm(crop.image);
    final List<OrtValue?> rtmOuts = await _run(_rtm!, {'input': rtmInput});

    final List<List<double>> kpts640 = _decodeAuto(rtmOuts, crop.meta);
    for (final OrtValue? v in rtmOuts) {
      v?.release();
    }

    final List<List<double>> coco = _toRawCoords(kpts640, lb.meta, image.width, image.height);
    final List<List<double>> h36m = _cocoToH36M(coco);

    return _ChainData(
      letterbox: lb.meta,
      bbox: det,
      coco: coco,
      h36m: h36m,
    );
  }

  Future<List<OrtValue?>> _run(OrtSession session, Map<String, OrtValue> inputs) async {
    final opts = OrtRunOptions();
    final outs = await session.runAsync(opts, inputs);
    opts.release();
    for (final entry in inputs.values) {
      entry.release();
    }
    return outs ?? const [];
  }

  _LetterboxResult _letterbox(img.Image src) {
    const int outSize = 640;
    final double r = math.min(outSize / src.width, outSize / src.height);
    final int newW = (src.width * r).round();
    final int newH = (src.height * r).round();

    final img.Image resized = img.copyResize(
      src,
      width: newW,
      height: newH,
      interpolation: img.Interpolation.cubic,
    );

    final img.Image canvas = img.Image(width: outSize, height: outSize);
    img.fill(canvas, color: img.ColorRgb8(114, 114, 114));
    final int dw = ((outSize - newW) / 2).floor();
    final int dh = ((outSize - newH) / 2).floor();
    img.compositeImage(canvas, resized, dstX: dw, dstY: dh);

    return _LetterboxResult(
      image: canvas,
      meta: _LetterboxMeta(
        inW: src.width,
        inH: src.height,
        outW: outSize,
        outH: outSize,
        r: r,
        dw: dw.toDouble(),
        dh: dh.toDouble(),
      ),
    );
  }

  OrtValue _prepYolo(img.Image im) {
    final int w = im.width;
    final int h = im.height;
    final int plane = w * h;
    final Float32List buf = Float32List(plane * 3);
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final img.Pixel pixel = im.getPixel(x, y);
        final int idx = y * w + x;
        buf[idx] = pixel.rNormalized.toDouble();
        buf[plane + idx] = pixel.gNormalized.toDouble();
        buf[(plane * 2) + idx] = pixel.bNormalized.toDouble();
      }
    }
    return OrtValueTensor.createTensorWithDataList(buf, [1, 3, h, w]);
  }

  OrtValue _prepRtm(img.Image patch) {
    final int w = patch.width;
    final int h = patch.height;
    final int plane = w * h;
    final Float32List buf = Float32List(plane * 3);
    const List<double> mean = [0.485, 0.456, 0.406];
    const List<double> std = [0.229, 0.224, 0.225];
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final img.Pixel pixel = patch.getPixel(x, y);
        final int idx = y * w + x;
        final double r = pixel.rNormalized.toDouble();
        final double g = pixel.gNormalized.toDouble();
        final double b = pixel.bNormalized.toDouble();
        buf[idx] = (r - mean[0]) / std[0];
        buf[plane + idx] = (g - mean[1]) / std[1];
        buf[(plane * 2) + idx] = (b - mean[2]) / std[2];
      }
    }
    return OrtValueTensor.createTensorWithDataList(buf, [1, 3, h, w]);
  }

  _Det? _pickLargestPerson(
    OrtValue output,
    _LetterboxMeta lb,
    int imgW,
    int imgH,
  ) {
    final List<int> shape = _shapeOf(output.value);
    if (shape.length != 3 || shape[0] != 1) {
      return null;
    }

    final List<List<double>> rows = _extractYoloRows(output.value, shape);
    if (rows.isEmpty) {
      return null;
    }

    final List<List<double>> boxes = [];
    final List<double> scores = [];
    for (final row in rows) {
      if (row.length < 84) continue;
      final double obj = row[4];
      final double cls = row[5];
      final double score = _sigmoid(obj) * _sigmoid(cls);
      if (score < 0.25) continue;

      final double cx = row[0];
      final double cy = row[1];
      final double w = row[2];
      final double h = row[3];
      final bool normalized =
          cx.abs() <= 2.0 && cy.abs() <= 2.0 && w.abs() <= 2.0 && h.abs() <= 2.0;
      final double scale = normalized ? lb.outW.toDouble() : 1.0;

      final double x1 = (cx - w / 2) * scale;
      final double y1 = (cy - h / 2) * scale;
      final double x2 = (cx + w / 2) * scale;
      final double y2 = (cy + h / 2) * scale;

      final double rawX1 = ((x1 - lb.dw) / lb.r).clamp(0.0, imgW.toDouble());
      final double rawY1 = ((y1 - lb.dh) / lb.r).clamp(0.0, imgH.toDouble());
      final double rawX2 = ((x2 - lb.dw) / lb.r).clamp(0.0, imgW.toDouble());
      final double rawY2 = ((y2 - lb.dh) / lb.r).clamp(0.0, imgH.toDouble());

      if (rawX2 <= rawX1 || rawY2 <= rawY1) continue;

      boxes.add([rawX1, rawY1, rawX2, rawY2]);
      scores.add(score);
    }

    if (boxes.isEmpty) {
      return null;
    }

    final List<int> keep = _nms(boxes, scores, 0.45, 50);
    if (keep.isEmpty) {
      return null;
    }

    int best = keep.first;
    double bestArea = _area(boxes[best]);
    double bestScore = scores[best];
    for (final idx in keep.skip(1)) {
      final double area = _area(boxes[idx]);
      final double score = scores[idx];
      if (area > bestArea || (area == bestArea && score > bestScore)) {
        best = idx;
        bestArea = area;
        bestScore = score;
      }
    }

    final List<double> b = boxes[best];
    return _Det(b[0], b[1], b[2], b[3], scores[best]);
  }

  _Det _rawToLb(
    _Det det,
    _LetterboxMeta lb, {
    bool clamp = true,
    double margin = 1.0,
  }) {
    final double cx = (det.x1 + det.x2) * 0.5;
    final double cy = (det.y1 + det.y2) * 0.5;
    double w = (det.x2 - det.x1) * margin;
    double h = (det.y2 - det.y1) * margin;
    if (w <= 1) w = 1;
    if (h <= 1) h = 1;

    double x1 = cx - w / 2;
    double y1 = cy - h / 2;
    double x2 = cx + w / 2;
    double y2 = cy + h / 2;

    x1 = x1 * lb.r + lb.dw;
    y1 = y1 * lb.r + lb.dh;
    x2 = x2 * lb.r + lb.dw;
    y2 = y2 * lb.r + lb.dh;

    if (clamp) {
      x1 = x1.clamp(0.0, lb.outW.toDouble());
      y1 = y1.clamp(0.0, lb.outH.toDouble());
      x2 = x2.clamp(0.0, lb.outW.toDouble());
      y2 = y2.clamp(0.0, lb.outH.toDouble());
    }

    return _Det(x1, y1, x2, y2, det.score);
  }

  _CropResult _cropForRtm(img.Image image, _Det det, {required int outH, required int outW}) {
    final double x1 = det.x1.clamp(0.0, image.width.toDouble());
    final double y1 = det.y1.clamp(0.0, image.height.toDouble());
    final double x2 = det.x2.clamp(0.0, image.width.toDouble());
    final double y2 = det.y2.clamp(0.0, image.height.toDouble());

    final img.Image cropped = img.copyCrop(
      image,
      x: x1.toInt(),
      y: y1.toInt(),
      width: math.max(1, (x2 - x1).toInt()),
      height: math.max(1, (y2 - y1).toInt()),
    );

    final img.Image resized = img.copyResize(
      cropped,
      width: outW,
      height: outH,
      interpolation: img.Interpolation.cubic,
    );

    final _CropMeta meta = _CropMeta(
      roiX: x1,
      roiY: y1,
      roiW: (x2 - x1),
      roiH: (y2 - y1),
      outW: outW,
      outH: outH,
    );

    return _CropResult(image: resized, meta: meta);
  }

  List<List<double>> _decodeAuto(List<OrtValue?> outs, _CropMeta meta) {
    if (outs.isEmpty || outs[0] == null) {
      return List.generate(17, (_) => [0.0, 0.0, 0.0]);
    }
    final List<int> shape0 = _shapeOf(outs[0]!.value);
    if (shape0.length == 4) {
      return _decodeHeatmap(outs[0]!, meta);
    }
    if (outs.length >= 2) {
      final List<int> shape1 = _shapeOf(outs[1]!.value);
      final bool looksSimcc =
          shape0.length == 3 && shape1.length == 3 && shape0[0] == 1 && shape1[0] == 1;
      if (looksSimcc) {
        final bool firstIsX = shape0.last <= shape1.last;
        final List<OrtValue?> ordered = firstIsX ? outs : [outs[1], outs[0]];
        return _decodeSimCC(ordered, meta);
      }
    }
    return _decodeHeatmap(outs[0]!, meta);
  }

  List<List<double>> _decodeHeatmap(OrtValue heat, _CropMeta meta) {
    final List data = heat.value as List;
    if (data.isEmpty) {
      return List.generate(17, (_) => [0.0, 0.0, 0.0]);
    }
    final List joints = data[0] as List;
    final int H = (joints[0] as List).length;
    final int W = ((joints[0] as List)[0] as List).length;
    final List<List<double>> result = List.generate(17, (_) => [0.0, 0.0, 0.0]);
    for (int j = 0; j < math.min(17, joints.length); j++) {
      final List plane = joints[j] as List;
      double best = -1e9;
      int bx = 0;
      int by = 0;
      for (int y = 0; y < H; y++) {
        final List row = plane[y] as List;
        for (int x = 0; x < W; x++) {
          final double v = (row[x] as num).toDouble();
          if (v > best) {
            best = v;
            bx = x;
            by = y;
          }
        }
      }
      final double px = (bx + 0.5) * (meta.outW / W);
      final double py = (by + 0.5) * (meta.outH / H);
      result[j][0] = meta.roiX + (px / meta.outW) * meta.roiW;
      result[j][1] = meta.roiY + (py / meta.outH) * meta.roiH;
      result[j][2] = 1.0;
    }
    return result;
  }

  List<List<double>> _decodeSimCC(List<OrtValue?> outs, _CropMeta meta) {
    final List<List<List<double>>> x = _toList3D(outs[0]!.value);
    final List<List<List<double>>> y = _toList3D(outs[1]!.value);
    final int joints = math.min(17, x[0].length);
    final List<List<double>> result = List.generate(17, (_) => [0.0, 0.0, 0.0]);
    const double split = 2.0;
    for (int j = 0; j < joints; j++) {
      final List<double> rowX = x[0][j];
      final List<double> rowY = y[0][j];
      int ix = 0;
      int iy = 0;
      double vx = -1e9;
      double vy = -1e9;
      for (int i = 0; i < rowX.length; i++) {
        if (rowX[i] > vx) {
          vx = rowX[i];
          ix = i;
        }
      }
      for (int i = 0; i < rowY.length; i++) {
        if (rowY[i] > vy) {
          vy = rowY[i];
          iy = i;
        }
      }
      final double px = ix / split;
      final double py = iy / split;
      result[j][0] = meta.roiX + (px / meta.outW) * meta.roiW;
      result[j][1] = meta.roiY + (py / meta.outH) * meta.roiH;
      result[j][2] = math.min(1.0, math.max(0.0, (vx + vy) * 0.5));
    }
    return result;
  }

  List<List<List<double>>> _toList3D(dynamic v) =>
      (v as List)
          .map<List<List<double>>>((a) =>
              (a as List)
                  .map<List<double>>((b) =>
                      (b as List).map<double>((c) => (c as num).toDouble()).toList())
                  .toList())
          .toList();

  List<int> _shapeOf(dynamic v) {
    final List<int> dims = [];
    dynamic cur = v;
    while (cur is List) {
      dims.add(cur.length);
      cur = cur.isNotEmpty ? cur[0] : null;
    }
    return dims;
  }

  List<List<double>> _toRawCoords(
    List<List<double>> kpts,
    _LetterboxMeta lb,
    int imgW,
    int imgH,
  ) {
    final List<List<double>> result =
        List.generate(kpts.length, (_) => [0.0, 0.0, 0.0]);
    for (int i = 0; i < kpts.length; i++) {
      final double x = ((kpts[i][0] - lb.dw) / lb.r).clamp(0.0, imgW.toDouble());
      final double y = ((kpts[i][1] - lb.dh) / lb.r).clamp(0.0, imgH.toDouble());
      result[i][0] = x;
      result[i][1] = y;
      result[i][2] = kpts[i][2];
    }
    return result;
  }

  List<List<double>> _cocoToH36M(List<List<double>> coco) {
    double px(int i) => coco[i][0];
    double py(int i) => coco[i][1];
    double pc(int i) => coco[i][2];

    final List<double> lhip = [px(11), py(11)];
    final List<double> rhip = [px(12), py(12)];
    final List<double> pelvis = [(lhip[0] + rhip[0]) / 2.0, (lhip[1] + rhip[1]) / 2.0];
    final List<double> lsho = [px(5), py(5)];
    final List<double> rsho = [px(6), py(6)];
    final List<double> neck = [(lsho[0] + rsho[0]) / 2.0, (lsho[1] + rsho[1]) / 2.0];
    final List<double> spine1 = [(pelvis[0] + neck[0]) / 2.0, (pelvis[1] + neck[1]) / 2.0];
    final List<double> nose = [px(0), py(0)];
    final List<double> leye = [px(1), py(1)];
    final List<double> reye = [px(2), py(2)];
    final bool hasEyes = pc(1) > 0 && pc(2) > 0;
    final List<double> head = hasEyes
        ? [(leye[0] + reye[0]) / 2.0, (leye[1] + reye[1]) / 2.0]
        : [nose[0], nose[1]];
    final List<List<double>> h36m = List.generate(17, (_) => [0.0, 0.0, 0.0]);

    void setJoint(int idx, double x, double y, double c) {
      h36m[idx][0] = x;
      h36m[idx][1] = y;
      h36m[idx][2] = c;
    }

    final double cPelvis = (pc(11) + pc(12)) / 2.0;
    final double cNeck = (pc(5) + pc(6)) / 2.0;
    final double cSpine1 = (cPelvis + cNeck) / 2.0;
    final double cHead = hasEyes ? (pc(1) + pc(2)) / 2.0 : pc(0);

    setJoint(0, pelvis[0], pelvis[1], cPelvis);
    setJoint(1, px(12), py(12), pc(12));
    setJoint(2, px(14), py(14), pc(14));
    setJoint(3, px(16), py(16), pc(16));
    setJoint(4, px(11), py(11), pc(11));
    setJoint(5, px(13), py(13), pc(13));
    setJoint(6, px(15), py(15), pc(15));
    setJoint(7, spine1[0], spine1[1], cSpine1);
    setJoint(8, neck[0], neck[1], cNeck);
    setJoint(9, head[0], head[1], cHead);
    setJoint(10, nose[0], nose[1], pc(0));
    setJoint(11, px(5), py(5), pc(5));
    setJoint(12, px(7), py(7), pc(7));
    setJoint(13, px(9), py(9), pc(9));
    setJoint(14, px(6), py(6), pc(6));
    setJoint(15, px(8), py(8), pc(8));
    setJoint(16, px(10), py(10), pc(10));

    return h36m;
  }

  List<List<double>> _extractYoloRows(dynamic value, List<int> shape) {
    if (shape[1] == 84) {
      final List channels = (value as List)[0] as List;
      final int count = shape[2];
      return List.generate(count, (i) {
        final List<double> row = List.filled(84, 0.0);
        for (int c = 0; c < 84; c++) {
          row[c] = ((channels[c] as List)[i] as num).toDouble();
        }
        return row;
      });
    }
    if (shape[2] == 84) {
      return ((value as List)[0] as List)
          .map<List<double>>((row) =>
              (row as List).map<double>((e) => (e as num).toDouble()).toList())
          .toList();
    }
    return const [];
  }

  double _sigmoid(double x) => 1 / (1 + math.exp(-x));

  List<int> _nms(
    List<List<double>> boxes,
    List<double> scores,
    double iouThr,
    int maxDet,
  ) {
    final List<int> order = List<int>.generate(scores.length, (i) => i)
      ..sort((a, b) => scores[b].compareTo(scores[a]));
    final List<int> keep = [];
    while (order.isNotEmpty && keep.length < maxDet) {
      final int i = order.removeAt(0);
      keep.add(i);
      order.removeWhere((j) => _iou(boxes[i], boxes[j]) > iouThr);
    }
    return keep;
  }

  double _iou(List<double> a, List<double> b) {
    final double x1 = math.max(a[0], b[0]);
    final double y1 = math.max(a[1], b[1]);
    final double x2 = math.min(a[2], b[2]);
    final double y2 = math.min(a[3], b[3]);
    final double interW = math.max(0.0, x2 - x1);
    final double interH = math.max(0.0, y2 - y1);
    final double inter = interW * interH;
    if (inter <= 0) return 0.0;
    final double areaA = _area(a);
    final double areaB = _area(b);
    final double union = areaA + areaB - inter;
    if (union <= 0) return 0.0;
    return inter / union;
  }

  double _area(List<double> b) => math.max(0.0, b[2] - b[0]) * math.max(0.0, b[3] - b[1]);
}

class Pose2DRecord {
  Pose2DRecord({
    required this.t,
    required this.tsMs,
    required this.frameIndex,
    required this.frameSize,
    required this.bbox,
    required this.letterbox,
    required this.keypoints,
  });

  final double t;
  final int tsMs;
  final int frameIndex;
  final List<int> frameSize;
  final _Det? bbox;
  final _LetterboxMeta letterbox;
  final List<List<double>> keypoints;

  Map<String, dynamic> toJson() {
    final double w = frameSize[0].toDouble();
    final double h = frameSize[1].toDouble();
    final List<double> bboxNorm;
    if (bbox == null) {
      bboxNorm = [0.0, 0.0, 0.0, 0.0];
    } else {
      bboxNorm = [
        (bbox!.x1 / w).clamp(0.0, 1.0),
        (bbox!.y1 / h).clamp(0.0, 1.0),
        (bbox!.x2 / w).clamp(0.0, 1.0),
        (bbox!.y2 / h).clamp(0.0, 1.0),
      ].map((v) => double.parse(v.toStringAsFixed(6))).toList();
    }

    return {
      't': double.parse(t.toStringAsFixed(3)),
      'ts_ms': tsMs,
      'frame_idx': frameIndex,
      'frame_size': frameSize,
      'bbox_norm': bboxNorm,
      'lb_params': {
        'r': double.parse(letterbox.r.toStringAsFixed(6)),
        'dw': double.parse(letterbox.dw.toStringAsFixed(3)),
        'dh': double.parse(letterbox.dh.toStringAsFixed(3)),
        'in_w': letterbox.inW,
        'in_h': letterbox.inH,
        'out': [letterbox.outW, letterbox.outH],
      },
      'kpt_coco': keypoints
          .map((kp) => [
                double.parse(kp[0].toStringAsFixed(3)),
                double.parse(kp[1].toStringAsFixed(3)),
                double.parse(kp[2].toStringAsFixed(3)),
              ])
          .toList(),
    };
  }
}

class _ChainData {
  _ChainData({
    required this.letterbox,
    required this.bbox,
    required this.coco,
    required this.h36m,
  });

  final _LetterboxMeta letterbox;
  final _Det? bbox;
  final List<List<double>> coco;
  final List<List<double>> h36m;
}

class _Det {
  _Det(this.x1, this.y1, this.x2, this.y2, this.score);

  final double x1;
  final double y1;
  final double x2;
  final double y2;
  final double score;
}

class _LetterboxMeta {
  _LetterboxMeta({
    required this.inW,
    required this.inH,
    required this.outW,
    required this.outH,
    required this.r,
    required this.dw,
    required this.dh,
  });

  final int inW;
  final int inH;
  final int outW;
  final int outH;
  final double r;
  final double dw;
  final double dh;
}

class _LetterboxResult {
  _LetterboxResult({required this.image, required this.meta});

  final img.Image image;
  final _LetterboxMeta meta;
}

class _CropMeta {
  _CropMeta({
    required this.roiX,
    required this.roiY,
    required this.roiW,
    required this.roiH,
    required this.outW,
    required this.outH,
  });

  final double roiX;
  final double roiY;
  final double roiW;
  final double roiH;
  final int outW;
  final int outH;
}

class _CropResult {
  _CropResult({required this.image, required this.meta});

  final img.Image image;
  final _CropMeta meta;
}
