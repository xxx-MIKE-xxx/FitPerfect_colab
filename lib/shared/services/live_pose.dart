// Live camera pose engine for streaming YOLO → RTMPose on a background isolate.
//
// This implementation follows the "Option B" design where we operate directly on
// the live camera stream (no intermediate video recording). Frames are sampled
// according to a configurable stride, processed on a worker isolate, and the
// overlay/UI is updated via a throttled broadcast stream. Each processed frame
// produces a JSONL line which is written to disk; when capture stops, the JSONL
// is finalized and the MotionBERT stage can be invoked immediately.

import 'dart:async';
import 'dart:convert';
import 'dart:isolate';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

import 'ort_session.dart';
import 'yuv_converter.dart';

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
    this.padColor = const [114, 114, 114],
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

  LivePoseOptions copyWith({
    int? yoloInput,
    List<int>? padColor,
    double? personConf,
    double? personIou,
    int? personClassId,
    String? yoloUnits,
    String? yoloCoords,
    int? rtmH,
    int? rtmW,
    String? rtmPreproc,
    double? simccRatio,
    int? simccX,
    int? simccY,
    double? cropScale,
  }) {
    return LivePoseOptions(
      yoloInput: yoloInput ?? this.yoloInput,
      padColor: padColor ?? this.padColor,
      personConf: personConf ?? this.personConf,
      personIou: personIou ?? this.personIou,
      personClassId: personClassId ?? this.personClassId,
      yoloUnits: yoloUnits ?? this.yoloUnits,
      yoloCoords: yoloCoords ?? this.yoloCoords,
      rtmH: rtmH ?? this.rtmH,
      rtmW: rtmW ?? this.rtmW,
      rtmPreproc: rtmPreproc ?? this.rtmPreproc,
      simccRatio: simccRatio ?? this.simccRatio,
      simccX: simccX ?? this.simccX,
      simccY: simccY ?? this.simccY,
      cropScale: cropScale ?? this.cropScale,
    );
  }
}

/// Overlay data for the preview layer.
class OverlayData {
  final int frameIndex;
  final List<double> bbox;
  final List<List<double>> keypoints;
  final double confidence;
  final int timestampUs;

  const OverlayData({
    required this.frameIndex,
    required this.bbox,
    required this.keypoints,
    required this.confidence,
    required this.timestampUs,
  });
}

/// Callback invoked after the JSONL is finalized so the caller can trigger
/// MotionBERT or any downstream processing. Returning a Future allows callers to
/// chain additional async work.
typedef MotionBertCallback = Future<void> Function(
  String sessionId,
  String sessionDirectory,
  String jsonlPath,
);

/// Configuration for the ONNX Runtime models the worker should load.
class LivePoseModelConfig {
  final String yoloAssetPath;
  final String rtmAssetPath;
  final List<String> yoloProviders;
  final List<String> rtmProviders;

  const LivePoseModelConfig({
    required this.yoloAssetPath,
    required this.rtmAssetPath,
    this.yoloProviders = const ['coreml', 'nnapi', 'xnnpack', 'cpu'],
    this.rtmProviders = const ['coreml', 'nnapi', 'xnnpack', 'cpu'],
  });
}

/// High level controller that manages the camera stream, a worker isolate for
/// inference, and JSONL output.
class LivePoseEngine {
  LivePoseEngine({
    required CameraController cameraController,
    required LivePoseOptions options,
    required LivePoseModelConfig modelConfig,
    MotionBertCallback? onMotionBert,
  })  : _cameraController = cameraController,
        _options = options,
        _modelConfig = modelConfig,
        _motionBertCallback = onMotionBert;

  final CameraController _cameraController;
  final LivePoseOptions _options;
  final LivePoseModelConfig _modelConfig;
  final MotionBertCallback? _motionBertCallback;

  final StreamController<OverlayData> _overlayController =
      StreamController<OverlayData>.broadcast();

  Stream<OverlayData> get overlayStream => _overlayController.stream;

  IOSink? _jsonlSink;
  String? _jsonlPath;
  Directory? _sessionDir;
  String? _sessionId;

  bool _isStreaming = false;
  bool _processingFrame = false;
  int _frameStride = 3;
  int _frameCounter = 0;
  Duration _overlayInterval = const Duration(milliseconds: 66);
  DateTime _lastOverlayEmit = DateTime.fromMillisecondsSinceEpoch(0);
  void Function(Object error)? _onError;

  ReceivePort? _workerPort;
  ReceivePort? _workerErrorPort;
  ReceivePort? _workerExitPort;
  StreamSubscription? _workerSubscription;
  StreamSubscription? _workerErrorSubscription;
  SendPort? _workerSendPort;
  Completer<void>? _workerExitCompleter;

  /// Starts the camera image stream and worker isolate. Returns the session ID
  /// (directory name).
  Future<String> start({
    int frameStride = 3,
    int overlayFps = 15,
    bool writeToDocuments = true,
    void Function(Object error)? onError,
  }) async {
    if (_isStreaming) {
      return _sessionId!;
    }

    _frameStride = math.max(1, frameStride);
    _overlayInterval = overlayFps <= 0
        ? Duration.zero
        : Duration(milliseconds: (1000 / overlayFps).round());
    _onError = onError;

    final baseDir = writeToDocuments
        ? await getApplicationDocumentsDirectory()
        : await getTemporaryDirectory();
    final sessionId = DateTime.now().millisecondsSinceEpoch.toString();
    final sessionDir = Directory(p.join(baseDir.path, 'FitPerfect', sessionId));
    if (!await sessionDir.exists()) {
      await sessionDir.create(recursive: true);
    }
    final jsonlFile = File(p.join(sessionDir.path, 'coco_2d.jsonl'));
    if (await jsonlFile.exists()) {
      await jsonlFile.delete();
    }
    _jsonlSink = jsonlFile.openWrite(mode: FileMode.writeOnlyAppend);
    _jsonlPath = jsonlFile.path;
    _sessionDir = sessionDir;
    _sessionId = sessionId;
    _frameCounter = 0;
    _processingFrame = false;
    _lastOverlayEmit = DateTime.fromMillisecondsSinceEpoch(0);

    await _startWorker();
    await _cameraController.startImageStream(_onCameraFrame);

    _isStreaming = true;
    return sessionId;
  }

  /// Stops processing, finalizes JSONL, runs MotionBERT if configured, and
  /// returns the JSONL path.
  Future<String> stop() async {
    if (!_isStreaming) {
      return _jsonlPath ?? '';
    }
    _isStreaming = false;

    try {
      await _cameraController.stopImageStream();
    } catch (_) {
      // Camera might already be stopped; ignore.
    }

    if (_workerSendPort != null) {
      _workerSendPort!.send(const {'type': 'stop'});
    }
    await _workerExitCompleter?.future;
    await _disposeWorker();

    await _jsonlSink?.flush();
    await _jsonlSink?.close();
    _jsonlSink = null;
    final jsonlPath = _jsonlPath ?? '';

    if (_motionBertCallback != null && _sessionId != null && _sessionDir != null) {
      try {
        await _motionBertCallback!(_sessionId!, _sessionDir!.path, jsonlPath);
      } catch (err) {
        _onError?.call(err);
      }
    }

    return jsonlPath;
  }

  Future<void> dispose() async {
    if (_isStreaming) {
      await stop();
    }
    await _disposeWorker();
    await _overlayController.close();
  }

  Future<void> _startWorker() async {
    final workerPort = ReceivePort();
    final errorPort = ReceivePort();
    final exitPort = ReceivePort();
    final ready = Completer<void>();

    _workerPort = workerPort;
    _workerErrorPort = errorPort;
    _workerExitPort = exitPort;
    _workerExitCompleter = Completer<void>();

    _workerErrorSubscription = errorPort.listen((dynamic message) {
      if (message is List && message.length >= 2) {
        final error = message[0];
        final stack = message[1];
        if (kDebugMode) {
          debugPrint('[LivePoseEngine] Worker error: $error\n$stack');
        }
        _onError?.call(error);
      }
    });

    exitPort.listen((_) {
      _workerExitCompleter?.complete();
    });

    _workerSubscription = workerPort.listen((dynamic message) {
      if (message is Map && message['type'] == 'ready') {
        _workerSendPort = message['port'] as SendPort;
        ready.complete();
        return;
      }
      if (message is Map && message['type'] == 'frameResult') {
        _handleFrameResult(message);
      } else if (message is Map && message['type'] == 'error') {
        _processingFrame = false;
        final err = message['error'];
        _onError?.call(err is Object ? err : Exception(err.toString()));
      }
    });

    final initMessage = _WorkerInitMessage(
      sendPort: workerPort.sendPort,
      options: _options,
      modelConfig: _modelConfig,
    );

    await Isolate.spawn<_WorkerInitMessage>(
      _poseWorkerMain,
      initMessage,
      onError: errorPort.sendPort,
      onExit: exitPort.sendPort,
    );

    await ready.future;
  }

  Future<void> _disposeWorker() async {
    await _workerSubscription?.cancel();
    await _workerErrorSubscription?.cancel();
    _workerPort?.close();
    _workerErrorPort?.close();
    _workerExitPort?.close();
    _workerSubscription = null;
    _workerErrorSubscription = null;
    _workerPort = null;
    _workerErrorPort = null;
    _workerExitPort = null;
    _workerSendPort = null;
    _workerIsolate = null;
  }

  void _onCameraFrame(CameraImage image) {
    if (!_isStreaming || _workerSendPort == null) {
      return;
    }
    if (_processingFrame) {
      return; // Drop frame when worker is busy (latest-wins policy).
    }
    final currentIndex = _frameCounter++;
    if ((currentIndex % _frameStride) != 0) {
      return;
    }

    _processingFrame = true;

    Future.microtask(() async {
      try {
        final rgbImage = yuv420ToImage(image);
        final rgbBytes = imageToRgbBytes(rgbImage);
        final payload = <String, Object?>{
          'type': 'frame',
          'frameIndex': currentIndex,
          'width': rgbImage.width,
          'height': rgbImage.height,
          'timestamp': DateTime.now().microsecondsSinceEpoch,
          'rgb': TransferableTypedData.fromList([rgbBytes]),
        };
        _workerSendPort?.send(payload);
      } catch (err) {
        _processingFrame = false;
        _onError?.call(err is Object ? err : Exception(err.toString()));
      }
    });
  }

  void _handleFrameResult(Map<dynamic, dynamic> message) {
    _processingFrame = false;
    final jsonLine = message['jsonl'] as String?;
    if (jsonLine != null) {
      _jsonlSink?.writeln(jsonLine);
    }

    final now = DateTime.now();
    if (_overlayController.isClosed) {
      return;
    }

    if (_overlayInterval == Duration.zero ||
        now.difference(_lastOverlayEmit) >= _overlayInterval) {
      _lastOverlayEmit = now;
      final overlay = OverlayData(
        frameIndex: message['frameIndex'] as int? ?? 0,
        bbox: (message['bbox'] as List?)?.cast<double>() ?? const [],
        keypoints: (message['keypoints'] as List?)
                ?.map<List<double>>((dynamic row) =>
                    (row as List).map((e) => (e as num).toDouble()).toList())
                .toList() ??
            const [],
        confidence: (message['confidence'] as num?)?.toDouble() ?? 0.0,
        timestampUs: message['timestamp'] as int? ?? 0,
      );
      _overlayController.add(overlay);
    }
  }

}

/// Lightweight helper used by the exercise preview screen to synchronously
/// process individual camera frames on the UI isolate. It reuses the same
/// ONNX models as the full [LivePoseEngine] but exposes a much simpler API
/// focused on `processFrame` → `LivePoseFrame` results.
class LivePosePreviewEngine {
  LivePosePreviewEngine({
    this.yoloFrameInterval = 1,
    double roiMargin = 1.25,
    LivePoseOptions? options,
    LivePoseModelConfig? modelConfig,
    this.confidenceThreshold = 0.1,
    bool? assumeNv21,
  })  : _options = (options ?? const LivePoseOptions())
            .copyWith(cropScale: roiMargin),
        _modelConfig = modelConfig ??
            const LivePoseModelConfig(
              yoloAssetPath: 'assets/models/yolov8n.onnx',
              rtmAssetPath: 'assets/models/rtmpose-m_256x192.onnx',
            ),
        _assumeNv21 = assumeNv21;

  final int yoloFrameInterval;
  final LivePoseOptions _options;
  final LivePoseModelConfig _modelConfig;
  final bool? _assumeNv21;
  final double confidenceThreshold;

  LivePose? _live;
  OrtSession? _yoloSession;
  OrtSession? _rtmSession;
  Future<void>? _initFuture;
  double? _lastConfidence;

  double? get lastConfidence => _lastConfidence;

  Future<void> ensureInitialized() async {
    if (_live != null) return;
    _initFuture ??= _load();
    try {
      await _initFuture;
    } catch (error) {
      _initFuture = null;
      rethrow;
    }
  }

  Future<LivePoseFrame?> processFrame(CameraImage image) async {
    await ensureInitialized();
    final live = _live;
    if (live == null) return null;

    final rgbImage = yuv420ToImage(image, assumeNV21: _assumeNv21);
    final rgbBytes = imageToRgbBytes(rgbImage);
    final frame = await live.processFrameRgb(
      rgbBytes,
      rgbImage.width,
      rgbImage.height,
    );
    _lastConfidence = frame.confMean;
    if (frame.confMean < confidenceThreshold) {
      return null;
    }
    return frame;
  }

  void dispose() {
    try {
      _yoloSession?.release();
    } catch (_) {}
    try {
      _rtmSession?.release();
    } catch (_) {}
    _yoloSession = null;
    _rtmSession = null;
    _live = null;
    _initFuture = null;
    _lastConfidence = null;
  }

  Future<void> _load() async {
    final yoloSession = await OrtManager.fromAssetWithProviders(
      _modelConfig.yoloAssetPath,
      providers: _modelConfig.yoloProviders,
    );
    final rtmSession = await OrtManager.fromAssetWithProviders(
      _modelConfig.rtmAssetPath,
      providers: _modelConfig.rtmProviders,
    );

    try {
      _yoloSession = yoloSession;
      _rtmSession = rtmSession;
      _live = await LivePose.create(
        options: _options,
        yoloRun: _runYolo,
        rtmRun: _runRtm,
      );
    } catch (error) {
      try {
        yoloSession.release();
      } catch (_) {}
      try {
        rtmSession.release();
      } catch (_) {}
      rethrow;
    }
  }

  Future<Float32List> _runYolo(Float32List chwInput) async {
    final session = _yoloSession;
    if (session == null) return Float32List(0);
    final input = OrtValueTensor.createTensorWithDataList(
      chwInput,
      [1, 3, _options.yoloInput, _options.yoloInput],
    );
    final runOptions = OrtRunOptions();
    final outputs = await session.runAsync(runOptions, {'images': input});
    runOptions.release();
    input.release();
    if (outputs == null || outputs.isEmpty) {
      return Float32List(0);
    }
    final first = outputs.first!;
    final flattened = _ortValueToFloat32List(first);
    for (final value in outputs) {
      value?.release();
    }
    return flattened;
  }

  Future<Map<String, Float32List>> _runRtm(Float32List chwInput) async {
    final session = _rtmSession;
    if (session == null) return <String, Float32List>{};
    final input = OrtValueTensor.createTensorWithDataList(
      chwInput,
      [1, 3, _options.rtmH, _options.rtmW],
    );
    final runOptions = OrtRunOptions();
    final outputs = await session.runAsync(runOptions, {'input': input});
    runOptions.release();
    input.release();
    final result = <String, Float32List>{};
    if (outputs != null) {
      final names = session.outputNames;
      for (int i = 0; i < outputs.length; i++) {
        final value = outputs[i];
        if (value == null) continue;
        final name = (names.length > i) ? names[i] : 'output_$i';
        result[name] = _ortValueToFloat32List(value);
      }
      for (final value in outputs) {
        value?.release();
      }
      result['simcc_x'] ??= result['output_0'] ?? result['0'] ?? Float32List(0);
      result['simcc_y'] ??= result['output_1'] ?? result['1'] ?? Float32List(0);
    }
    return result;
  }
}

Uint8List imageToRgbBytes(img.Image image) {
  final out = Uint8List(image.width * image.height * 3);
  int offset = 0;
  for (int y = 0; y < image.height; y++) {
    for (int x = 0; x < image.width; x++) {
      final pixel = image.getPixel(x, y);
      out[offset++] = img.getRed(pixel);
      out[offset++] = img.getGreen(pixel);
      out[offset++] = img.getBlue(pixel);
    }
  }
  return out;
}

/// YOLO inference callback signature.
/// Must return the raw detection tensor as flattened Float32List of shape [1,84,N].
typedef YoloRun = Future<Float32List> Function(Float32List chwInput);

/// RTMPose inference callback signature.
/// Must return {'simcc_x': Float32List[1,17,Wx], 'simcc_y': Float32List[1,17,Hy]}.
typedef RtmRun = Future<Map<String, Float32List>> Function(Float32List chwInput);

/// Per-frame output for overlay and logging.
class LivePoseFrame {
  final int t;
  /// Bounding box [x1,y1,x2,y2] in image pixels.
  final List<double> bbox;
  /// COCO-17 keypoints [[x,y,conf] * 17], image pixel coords.
  final List<List<double>> kptsCoco;
  /// Mean keypoint confidence for quick QA.
  final double confMean;
  /// Serialized JSONL line describing the frame.
  final String jsonl;

  LivePoseFrame(this.t, this.bbox, this.kptsCoco, this.confMean, this.jsonl);
}

class LivePose {
  LivePose._(this.options, this._yoloRun, this._rtmRun);
  final LivePoseOptions options;
  final YoloRun _yoloRun;
  final RtmRun _rtmRun;

  int _t = 0;
  List<double>? _prevBbox; // prev detection for stability

  static Future<LivePose> create({
    required LivePoseOptions options,
    required YoloRun yoloRun,
    required RtmRun rtmRun,
  }) async {
    final live = LivePose._(options, yoloRun, rtmRun);
    live._t = 0;
    live._prevBbox = null;
    return live;
  }

  void reset() {
    _t = 0;
    _prevBbox = null;
  }

  /// Process one RGB frame (packed RGB uint8, HxWx3).
  Future<LivePoseFrame> processFrameRgb(
      Uint8List rgb, int width, int height) async {
    // 1) Letterbox to [640,640], keep r, dw, dh
    final lb = _letterbox(
      rgb,
      width,
      height,
      options.yoloInput,
      options.yoloInput,
      pad: options.padColor,
    );

    // 2) YOLO input [1,3,640,640] float32 normalized depending on preproc
    final yoloIn = _rgbToCHWFloat(
      lb.image,
      lb.canvasW,
      lb.canvasH,
      divideBy255: true,
    );
    final yoloOut = await _yoloRun(yoloIn); // flattened [1,84,N]
    final int n = yoloOut.isEmpty ? 0 : yoloOut.length ~/ 84;
    if (n == 0) {
      // no detections was returned; fallback
      final bbox = _fallbackBox(width, height);
      final crop = _cropToAspect(
        rgb,
        width,
        height,
        bbox,
        options.rtmH,
        options.rtmW,
        scale: options.cropScale,
      );
      final rtmIn = _rgbToCHWFloat(
        crop.image,
        options.rtmW,
        options.rtmH,
        divideBy255: options.rtmPreproc.endsWith('255'),
      );
      final simcc = await _rtmRun(rtmIn);
      final tensors = _requireSimcc(simcc);
      final coords = _simccDecode(
        tensors.x,
        tensors.y,
        options.simccRatio,
        options.simccX,
        options.simccY,
      );
      final coco = _coordsToImage(coords, crop.rect, options.rtmW, options.rtmH);
      final confMean = _mean(coco.map((e) => e[2]).toList());
      final jsonl = _encodeJsonl(
        _t,
        bbox,
        confMean,
        coco,
        imgW: width,
        imgH: height,
      );
      final frame = LivePoseFrame(_t, bbox, coco, confMean, jsonl);
      _t += 1;
      return frame;
    }

    // 3) Decode [1,84,N] → [N,84]
    final preds = _transpose84NToN84(yoloOut, n);

    // 4) Compute classes & scores
    const int C = 80; // COCO-80
    final scores = Float32List(n);
    final ids = Int32List(n);
    for (int i = 0; i < n; i++) {
      // logits → sigmoid for class scores
      double bestScore = -1.0;
      int bestId = -1;
      for (int c = 0; c < C; c++) {
        final v = _sigmoid(preds[i][4 + c]);
        if (v > bestScore) {
          bestScore = v;
          bestId = c;
        }
      }
      scores[i] = bestScore;
      ids[i] = bestId;
    }

    // 5) Keep person class and score threshold
    final keepPerson = <int>[];
    for (int i = 0; i < n; i++) {
      if (ids[i] == options.personClassId &&
          scores[i] >= options.personConf) keepPerson.add(i);
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
      final crop = _cropToAspect(
        rgb,
        width,
        height,
        bbox,
        options.rtmH,
        options.rtmW,
        scale: options.cropScale,
      );
      final rtmIn = _rgbToCHWFloat(
        crop.image,
        options.rtmW,
        options.rtmH,
        divideBy255: options.rtmPreproc.endsWith('255'),
      );
      final simcc = await _rtmRun(rtmIn);
      final tensors = _requireSimcc(simcc);
      final coords = _simccDecode(
        tensors.x,
        tensors.y,
        options.simccRatio,
        options.simccX,
        options.simccY,
      );
      final coco = _coordsToImage(coords, crop.rect, options.rtmW, options.rtmH);
      final confMean = _mean(coco.map((e) => e[2]).toList());
      final jsonl = _encodeJsonl(
        _t,
        bbox,
        confMean,
        coco,
        imgW: width,
        imgH: height,
      );
      final frame = LivePoseFrame(_t, bbox, coco, confMean, jsonl);
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
        boxesImg.add([
          _clip(x1, 0, width - 1),
          _clip(y1, 0, height - 1),
          _clip(x2, 0, width - 1),
          _clip(y2, 0, height - 1)
        ]);
      }
    }

    // 9) NMS
    final keep = _nms(boxesImg, scoresPerson, options.personIou, 50);
    if (keep.isEmpty) {
      final bbox = _prevBbox ?? _fallbackBox(width, height);
      final crop = _cropToAspect(
        rgb,
        width,
        height,
        bbox,
        options.rtmH,
        options.rtmW,
        scale: options.cropScale,
      );
      final rtmIn = _rgbToCHWFloat(
        crop.image,
        options.rtmW,
        options.rtmH,
        divideBy255: options.rtmPreproc.endsWith('255'),
      );
      final simcc = await _rtmRun(rtmIn);
      final tensors = _requireSimcc(simcc);
      final coords = _simccDecode(
        tensors.x,
        tensors.y,
        options.simccRatio,
        options.simccX,
        options.simccY,
      );
      final coco = _coordsToImage(coords, crop.rect, options.rtmW, options.rtmH);
      final confMean = _mean(coco.map((e) => e[2]).toList());
      final jsonl = _encodeJsonl(
        _t,
        bbox,
        confMean,
        coco,
        imgW: width,
        imgH: height,
      );
      final frame = LivePoseFrame(_t, bbox, coco, confMean, jsonl);
      _t += 1;
      return frame;
    }

    final best = keep.first;
    final bbox = boxesImg[best];
    _prevBbox = bbox;

    // 10) Crop→256x192 + RTMPose
    final crop = _cropToAspect(
      rgb,
      width,
      height,
      bbox,
      options.rtmH,
      options.rtmW,
      scale: options.cropScale,
    );
    final preDiv255 = options.rtmPreproc.endsWith('255');
    final rtmIn = _rgbToCHWFloat(
      crop.image,
      options.rtmW,
      options.rtmH,
      divideBy255: preDiv255,
    );
    final simcc = await _rtmRun(rtmIn);
    final tensors = _requireSimcc(simcc);

    // 11) SimCC decode and map back to image
    final coords = _simccDecode(
      tensors.x,
      tensors.y,
      options.simccRatio,
      options.simccX,
      options.simccY,
    );
    final coco = _coordsToImage(coords, crop.rect, options.rtmW, options.rtmH);
    final confMean = _mean(coco.map((e) => e[2]).toList());

    // 12) Save JSONL (normalized bbox like Python)
    final jsonl = _encodeJsonl(
      _t,
      bbox,
      confMean,
      coco,
      imgW: width,
      imgH: height,
    );

    final frame = LivePoseFrame(_t, bbox, coco, confMean, jsonl);
    _t += 1;
    return frame;
  }

  String _encodeJsonl(
    int t,
    List<double> bbox,
    double score,
    List<List<double>> coco, {
    int? imgW,
    int? imgH,
  }) {
    final w = (imgW ?? 1).toDouble();
    final h = (imgH ?? 1).toDouble();
    final line = {
      't': t,
      'bbox': [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h],
      'score': score,
      'yolo_output_units': options.yoloUnits,
      'yolo_output_coords': options.yoloCoords,
      'kpt_coco': coco,
    };
    return jsonEncode(line);
  }

  // ----------------- Internals -----------------

  double _clip(double v, double lo, double hi) =>
      v < lo
          ? lo
          : (v > hi ? hi : v);

  List<List<double>> _coordsToImage(
      List<List<double>> coordsIn, _Rect rect, int inW, int inH) {
    final rx = rect.x, ry = rect.y, rw = rect.w, rh = rect.h;
    final out = <List<double>>[];
    for (final p in coordsIn) {
      final x = rx + p[0] * (rw / inW);
      final y = ry + p[1] * (rh / inH);
      out.add([x, y, p[2]]);
    }
    return out;
  }

  List<List<double>> _simccDecode(
    Float32List simccX,
    Float32List simccY,
    double splitRatio,
    int Wx,
    int Hy,
  ) {
    // Shapes: [1,17,Wx], [1,17,Hy]
    const K = 17;
    final px = _softmaxLast(simccX, K, Wx);
    final py = _softmaxLast(simccY, K, Hy);
    final out = <List<double>>[];
    for (int k = 0; k < K; k++) {
      int argx = 0;
      double maxx = -1;
      for (int i = 0; i < Wx; i++) {
        final v = px[k * Wx + i];
        if (v > maxx) {
          maxx = v;
          argx = i;
        }
      }
      int argy = 0;
      double maxy = -1;
      for (int j = 0; j < Hy; j++) {
        final v = py[k * Hy + j];
        if (v > maxy) {
          maxy = v;
          argy = j;
        }
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

  List<int> _nms(
      List<List<double>> boxes, List<double> scores, double iouThr, int maxDet) {
    final idxs = List<int>.generate(boxes.length, (i) => i);
    idxs.sort((a, b) => scores[b].compareTo(scores[a]));
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
    for (final x in v) {
      s += x;
    }
    return s / v.length;
  }

  List<List<double>> _transpose84NToN84(Float32List flat84N, int N) {
    // Input shape [1,84,N]; ONNX is row-major (last dim contiguous).
    // So flat84N is laid out as 84 blocks of size N: [84][N].
    final out = List<List<double>>.generate(
      N,
      (_) => List<double>.filled(84, 0.0),
    );
    int idx = 0;
    for (int c = 0; c < 84; c++) {
      for (int i = 0; i < N; i++) {
        out[i][c] = flat84N[idx++];
      }
    }
    return out;
  }

  _LetterboxResult _letterbox(
    Uint8List rgb,
    int w,
    int h,
    int tw,
    int th, {
    List<int> pad = const [114, 114, 114],
  }) {
    final r = math.min(th / h, tw / w);
    final newW = (w * r).round();
    final newH = (h * r).round();
    final resized = _resizeRgbNearest(rgb, w, h, newW, newH);
    final dwf = (tw - newW) / 2.0;
    final dhf = (th - newH) / 2.0;
    final left = dwf.floor();
    final top = dhf.floor();
    final canvas = Uint8List(tw * th * 3);
    // fill pad color
    for (int i = 0; i < canvas.length; i += 3) {
      canvas[i + 0] = pad[0];
      canvas[i + 1] = pad[1];
      canvas[i + 2] = pad[2];
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

  Uint8List _resizeRgbNearest(
      Uint8List src, int sw, int sh, int dw, int dh) {
    final out = Uint8List(dw * dh * 3);
    final xRatio = sw / dw;
    final yRatio = sh / dh;
    for (int y = 0; y < dh; y++) {
      final sy = (y * yRatio).floor().clamp(0, sh - 1);
      for (int x = 0; x < dw; x++) {
        final sx = (x * xRatio).floor().clamp(0, sw - 1);
        final sIdx = (sy * sw + sx) * 3;
        final dIdx = (y * dw + x) * 3;
        out[dIdx] = src[sIdx];
        out[dIdx + 1] = src[sIdx + 1];
        out[dIdx + 2] = src[sIdx + 2];
      }
    }
    return out;
  }

  Float32List _rgbToCHWFloat(
    Uint8List rgb,
    int w,
    int h, {
    bool divideBy255 = true,
    List<double>? mean,
    List<double>? std,
  }) {
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
      for (int i = 0; i < h * w; i++) {
        out[offR + i] = ((out[offR + i] * 255.0) - mean[0]) / std[0];
        out[offG + i] = ((out[offG + i] * 255.0) - mean[1]) / std[1];
        out[offB + i] = ((out[offB + i] * 255.0) - mean[2]) / std[2];
      }
    }
    return out;
  }

  _CropResult _cropToAspect(
    Uint8List rgb,
    int w,
    int h,
    List<double> boxXYXY,
    int outH,
    int outW, {
    double scale = 1.25,
  }) {
    final x1 = boxXYXY[0], y1 = boxXYXY[1], x2 = boxXYXY[2], y2 = boxXYXY[3];
    final cx = (x1 + x2) / 2.0;
    final cy = (y1 + y2) / 2.0;
    double bw = (x2 - x1) * scale;
    double bh = (y2 - y1) * scale;

    final targetAr = outW / outH; // 192/256
    if (bw / bh > targetAr) {
      bh = bw / targetAr;
    } else {
      bw = bh * targetAr;
    }

    int rx1 = math.max(0, (cx - bw / 2.0).round());
    int ry1 = math.max(0, (cy - bh / 2.0).round());
    int rx2 = math.min(w - 1, (cx + bw / 2.0).round());
    int ry2 = math.min(h - 1, (cy + bh / 2.0).round());

    final rw = math.max(1, rx2 - rx1);
    final rh = math.max(1, ry2 - ry1);
    final crop = Uint8List(outW * outH * 3);
    for (int y = 0; y < outH; y++) {
      final sy = (ry1 + (y * rh / outH)).floor().clamp(0, h - 1);
      for (int x = 0; x < outW; x++) {
        final sx = (rx1 + (x * rw / outW)).floor().clamp(0, w - 1);
        final sIdx = (sy * w + sx) * 3;
        final dIdx = (y * outW + x) * 3;
        crop[dIdx] = rgb[sIdx];
        crop[dIdx + 1] = rgb[sIdx + 1];
        crop[dIdx + 2] = rgb[sIdx + 2];
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

  _SimccTensors _requireSimcc(Map<String, Float32List> tensors) {
    final simccX = tensors['simcc_x'] ?? tensors['output_0'];
    final simccY = tensors['simcc_y'] ?? tensors['output_1'];
    if (simccX == null || simccY == null) {
      throw StateError('RTMPose outputs missing simcc_x/simcc_y tensors');
    }
    return _SimccTensors(simccX, simccY);
  }
}

class _SimccTensors {
  final Float32List x;
  final Float32List y;
  const _SimccTensors(this.x, this.y);
}

class _LetterboxResult {
  final Uint8List image;
  final int canvasW;
  final int canvasH;
  final double r;
  final double dw;
  final double dh;
  _LetterboxResult({
    required this.image,
    required this.canvasW,
    required this.canvasH,
    required this.r,
    required this.dw,
    required this.dh,
  });
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

// ----------------- Worker isolate -----------------

class _WorkerInitMessage {
  final SendPort sendPort;
  final LivePoseOptions options;
  final LivePoseModelConfig modelConfig;

  const _WorkerInitMessage({
    required this.sendPort,
    required this.options,
    required this.modelConfig,
  });
}

void _poseWorkerMain(_WorkerInitMessage init) async {
  final controlPort = ReceivePort();
  init.sendPort.send({'type': 'ready', 'port': controlPort.sendPort});

  final options = init.options;
  final modelConfig = init.modelConfig;

  late final LivePose live;
  late final OrtSession yoloSession;
  late final OrtSession rtmSession;

  Future<Float32List> runYolo(Float32List chwInput) async {
    final input =
        OrtValueTensor.createTensorWithDataList(chwInput, [1, 3, options.yoloInput, options.yoloInput]);
    final runOptions = OrtRunOptions();
    final outputs = await yoloSession.runAsync(runOptions, {'images': input});
    runOptions.release();
    input.release();
    if (outputs == null || outputs.isEmpty) {
      return Float32List(0);
    }
    final out = outputs.first!;
    final flattened = _ortValueToFloat32List(out);
    for (final value in outputs) {
      value?.release();
    }
    return flattened;
  }

  Future<Map<String, Float32List>> runRtm(Float32List chwInput) async {
    final input =
        OrtValueTensor.createTensorWithDataList(chwInput, [1, 3, options.rtmH, options.rtmW]);
    final runOptions = OrtRunOptions();
    final outputs = await rtmSession.runAsync(runOptions, {'input': input});
    runOptions.release();
    input.release();
    final result = <String, Float32List>{};
    if (outputs != null) {
      final names = rtmSession.outputNames;
      for (int i = 0; i < outputs.length; i++) {
        final value = outputs[i];
        if (value == null) continue;
        final name = (names.length > i) ? names[i] : 'output_$i';
        result[name] = _ortValueToFloat32List(value);
      }
      for (final value in outputs) {
        value?.release();
      }
      result['simcc_x'] ??= result['output_0'] ?? result['0'];
      result['simcc_y'] ??= result['output_1'] ?? result['1'];
    }
    return result;
  }

  try {
    yoloSession = await OrtManager.fromAssetWithProviders(
      modelConfig.yoloAssetPath,
      providers: modelConfig.yoloProviders,
    );
    rtmSession = await OrtManager.fromAssetWithProviders(
      modelConfig.rtmAssetPath,
      providers: modelConfig.rtmProviders,
    );
    live = await LivePose.create(
      options: options,
      yoloRun: runYolo,
      rtmRun: runRtm,
    );
  } catch (err, stack) {
    init.sendPort.send({'type': 'error', 'error': err, 'stack': stack.toString()});
    controlPort.close();
    return;
  }

  await for (final dynamic message in controlPort) {
    if (message is Map && message['type'] == 'frame') {
      try {
        final TransferableTypedData data = message['rgb'] as TransferableTypedData;
        final Uint8List rgb = data.materialize().asUint8List();
        final frameIndex = message['frameIndex'] as int? ?? 0;
        final width = message['width'] as int? ?? 0;
        final height = message['height'] as int? ?? 0;
        final timestamp = message['timestamp'] as int? ?? 0;
        final frame = await live.processFrameRgb(rgb, width, height);
        init.sendPort.send({
          'type': 'frameResult',
          'frameIndex': frameIndex,
          'bbox': frame.bbox,
          'keypoints': frame.kptsCoco,
          'confidence': frame.confMean,
          'jsonl': frame.jsonl,
          'timestamp': timestamp,
        });
      } catch (err, stack) {
        init.sendPort.send({
          'type': 'error',
          'error': err,
          'stack': stack.toString(),
        });
      }
    } else if (message is Map && message['type'] == 'stop') {
      break;
    }
  }

  try {
    yoloSession.release();
  } catch (_) {}
  try {
    rtmSession.release();
  } catch (_) {}
  controlPort.close();
}

Float32List _ortValueToFloat32List(OrtValue value) {
  final dynamic raw = value.value;
  final length = _countElements(raw);
  final flat = Float32List(length);
  _fillFlat(raw, flat, 0);
  return flat;
}

int _countElements(dynamic value) {
  if (value is List) {
    int count = 0;
    for (final v in value) {
      count += _countElements(v);
    }
    return count;
  }
  if (value is num) {
    return 1;
  }
  return 0;
}

int _fillFlat(dynamic value, Float32List out, int offset) {
  if (value is List) {
    for (final v in value) {
      offset = _fillFlat(v, out, offset);
    }
    return offset;
  }
  if (value is num) {
    out[offset] = value.toDouble();
    return offset + 1;
  }
  return offset;
}

