// lib/shared/services/pose_processing_controller.dart
import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math';

import 'package:flutter/foundation.dart';

import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

import 'pose_runtime.dart';
import 'dart:ui' show Size;
import 'motionbert_runner.dart';

typedef SessionId = String;

class VideoMeta {
  VideoMeta({
    required this.sessionId,
    required this.fps,
    required this.width,
    required this.height,
    required this.exercise,
    required this.startTime,
    required this.endTime,
    required this.platform,
    required this.appVersion,
  });

  final SessionId sessionId;
  final double fps;
  final int width;
  final int height;
  final String exercise;
  final DateTime startTime;
  final DateTime endTime;
  final String platform;
  final String appVersion;
}

enum ProgressStatus { hidden, indeterminate, determinate, complete, error, cancelled }

class ProgressEvent {
  ProgressEvent._({
    required this.status,
    this.phase,
    this.value,
    this.processed,
    this.total,
    this.message,
    this.allowCancel = true,
  });

  final ProgressStatus status;
  final String? phase;
  final double? value;
  final int? processed;
  final int? total;
  final String? message;
  final bool allowCancel;

  bool get isHidden => status == ProgressStatus.hidden;

  factory ProgressEvent.hidden() =>
      ProgressEvent._(status: ProgressStatus.hidden, allowCancel: false);

  factory ProgressEvent.indeterminate({required String phase, bool allowCancel = true}) =>
      ProgressEvent._(status: ProgressStatus.indeterminate, phase: phase, allowCancel: allowCancel);

  factory ProgressEvent.determinate({
    required String phase,
    required double value,
    int? processed,
    int? total,
    bool allowCancel = true,
  }) =>
      ProgressEvent._(
        status: ProgressStatus.determinate,
        phase: phase,
        value: value,
        processed: processed,
        total: total,
        allowCancel: allowCancel,
      );

  factory ProgressEvent.complete({String phase = 'Completed'}) =>
      ProgressEvent._(status: ProgressStatus.complete, phase: phase, allowCancel: false, value: 1.0);

  factory ProgressEvent.error(String message) => ProgressEvent._(
        status: ProgressStatus.error,
        phase: 'Error',
        message: message,
        allowCancel: false,
      );

  factory ProgressEvent.cancelled() => ProgressEvent._(
        status: ProgressStatus.cancelled,
        phase: 'Cancelled',
        allowCancel: false,
      );
}

class PoseProcessingOutcome {
  PoseProcessingOutcome({
    required this.sessionId,
    required this.sessionDir,
    required this.sequence2d,
    required this.result3d,
    required this.summary2d,
    required this.summary3d,
  });

  final SessionId sessionId;
  final Directory sessionDir;
  final Pose2DSequence sequence2d;
  final Pose3DResult result3d;
  final Map<String, dynamic> summary2d;
  final Map<String, dynamic> summary3d;

  Map<String, dynamic> toFeedbackReport() => {
        'num_frames': sequence2d.frames.length,
        'num_joints': 17,
        'kpts2d': sequence2d.frames.map((f) => f.toRawList()).toList(),
        'kpts3d': result3d.sequence,
        'meta': {
          'fps': sequence2d.fps,
          'window': result3d.windowSize,
          'stride': result3d.stride,
        },
      };
}

class PoseProcessingCancelled implements Exception {
  const PoseProcessingCancelled();
}

class PoseProcessingController {
  PoseProcessingController({PosePipeline? pipeline}) : _pipeline = pipeline ?? PosePipeline();

  final PosePipeline _pipeline;
  // ─────────── New: MotionBERT on-device orchestrator (for live stream sessions) ───────────
  final MotionBertRunner _mbRunner = MotionBertRunner();
  bool _isProcessing3D = false;
  final _progressCtrl = StreamController<ProgressEvent>.broadcast();
  bool _processing = false;
  bool _cancelRequested = false;

  Stream<ProgressEvent> get progressStream => _progressCtrl.stream;
  bool get isProcessing => _processing;

  Future<PoseProcessingOutcome> processRecording({
    required File videoFile,
    required VideoMeta meta,
  }) async {
    if (_processing) {
      throw StateError('Processing already in progress');
    }

    _processing = true;
    _cancelRequested = false;

    Directory? sessionDir;

    try {
      sessionDir = await _prepareSessionDir(meta.sessionId);
      final dir2d = Directory(p.join(sessionDir.path, '2d'))..createSync(recursive: true);
      final dir3d = Directory(p.join(sessionDir.path, '3d'))..createSync(recursive: true);

      _progressCtrl.add(ProgressEvent.indeterminate(phase: 'Finalizing 2D frames'));
      final seq2d = await _pipeline.extract2D(
        videoFile,
        targetFps: meta.fps,
        shouldAbort: () => _cancelRequested,
      );

      if (_cancelRequested) {
        throw const PoseProcessingCancelled();
      }

      await _write2D(dir2d, seq2d);
      final summary2d = _build2DIndex(seq2d, meta);

      _progressCtrl.add(ProgressEvent.indeterminate(phase: 'Preparing 3D input'));

      final result3d = await _pipeline.estimate3D(
        seq2d,
        onProgress: (progress) {
          if (_cancelRequested) return;
          if (progress.phase == PosePipelinePhase.running3d) {
            _progressCtrl.add(
              ProgressEvent.determinate(
                phase: 'Estimating 3D poses',
                value: progress.fraction,
                processed: progress.processed,
                total: progress.total,
              ),
            );
          }
        },
        shouldAbort: () => _cancelRequested,
      );

      if (_cancelRequested) {
        throw const PoseProcessingCancelled();
      }

      await _write3D(dir3d, result3d);
      final summary3d = _build3DIndex(result3d);

      await _writeMeta(sessionDir, meta, summary2d, summary3d);

      _progressCtrl.add(ProgressEvent.complete(phase: '3D analysis complete'));

      return PoseProcessingOutcome(
        sessionId: meta.sessionId,
        sessionDir: sessionDir,
        sequence2d: seq2d,
        result3d: result3d,
        summary2d: summary2d,
        summary3d: summary3d,
      );
    } on PoseProcessingCancelled {
      if (sessionDir != null) {
        await _deleteDir(sessionDir);
      }
      _progressCtrl.add(ProgressEvent.cancelled());
      throw const PoseProcessingCancelled();
    } on PosePipelineCancelled {
      if (sessionDir != null) {
        await _deleteDir(sessionDir);
      }
      _progressCtrl.add(ProgressEvent.cancelled());
      throw const PoseProcessingCancelled();
    } catch (e, st) {
      if (sessionDir != null) {
        await _deleteDir(sessionDir);
      }
      if (kDebugMode) {
        debugPrint('Pose processing failed: $e\n$st');
      }
      _progressCtrl.add(ProgressEvent.error(e.toString()));
      rethrow;
    } finally {
      _processing = false;
      _cancelRequested = false;
    }
  }


  /// Runs MotionBERT 3D using the live 2D stream outputs (coco_2d.jsonl) written by LivePoseEngine.
  /// Returns the created out_3d.json file in Documents/FitPerfect/<sessionId>/, or null on failure.
  Future<File?> run3DForSession(
    String sessionId,
    Size frameSize, {
    bool rootRelative = true,
    bool writeNpy = false,
  }) async {
    if (_isProcessing3D) return null;
    _isProcessing3D = true;
    try {
      final file = await _mbRunner.run(
        sessionId: sessionId,
        frameSize: frameSize,
        rootRelative: rootRelative,
        writeNpy: writeNpy,
      );
      return file;
    } catch (e, st) {
      debugPrint('[PoseProcessingController] MotionBERT failed: $e');
      debugPrint('$st');
      return null;
    } finally {
      _isProcessing3D = false;
    }
  }

  void cancel() {
    if (!_processing) return;
    _cancelRequested = true;
  }

  Future<void> dispose() async {
    await _progressCtrl.close();
  }

  Future<Directory> _prepareSessionDir(SessionId sessionId) async {
    final base = await _baseDir();
    final sessionDir = Directory(p.join(base.path, 'poses', sessionId));
    if (sessionDir.existsSync()) {
      await sessionDir.delete(recursive: true);
    }
    await sessionDir.create(recursive: true);
    return sessionDir;
  }

  Future<Directory> _baseDir() async {
    final docs = await getApplicationDocumentsDirectory();
    final base = Directory(p.join(docs.path, 'FitPerfect'));
    if (!base.existsSync()) {
      await base.create(recursive: true);
    }
    return base;
  }

  Future<void> _write2D(Directory dir, Pose2DSequence seq) async {
    final framesPath = File(p.join(dir.path, 'frames.jsonl'));
    final tmp = File('${framesPath.path}.tmp');
    final sink = tmp.openWrite();
    for (final frame in seq.frames) {
      sink.writeln(jsonEncode(frame.toJson()));
    }
    await sink.flush();
    await sink.close();
    await tmp.rename(framesPath.path);

    final index = _build2DIndex(seq, null);
    await _writeJsonAtomic(File(p.join(dir.path, 'index.json')), index);
  }

  Future<void> _write3D(Directory dir, Pose3DResult result) async {
    final windowsPath = File(p.join(dir.path, 'windows.jsonl'));
    final tmp = File('${windowsPath.path}.tmp');
    final sink = tmp.openWrite();
    for (final window in result.windows) {
      sink.writeln(jsonEncode(window.toJson()));
    }
    await sink.flush();
    await sink.close();
    await tmp.rename(windowsPath.path);

    final index = _build3DIndex(result);
    await _writeJsonAtomic(File(p.join(dir.path, 'index.json')), index);
  }

  Future<void> _writeMeta(
    Directory dir,
    VideoMeta meta,
    Map<String, dynamic> index2d,
    Map<String, dynamic> index3d,
  ) async {
    final cfg = _pipeline.motionBertConfig;
    final payload = {
      'sessionId': meta.sessionId,
      'appVersion': meta.appVersion,
      'platform': meta.platform,
      'fps': meta.fps,
      'startTime': meta.startTime.toUtc().toIso8601String(),
      'endTime': meta.endTime.toUtc().toIso8601String(),
      'exercise': meta.exercise,
      'model2D': {
        'name': 'RTM-Pose rtm-m',
        'keypoints': 'RTM→H36M17',
        'confidenceMin': null,
      },
      'model3D': {
        'name': 'MotionBERT',
        'onnxPath': cfg.assetPath,
        'T': cfg.window,
      },
      'schemaVersion': 1,
      'index2d': index2d,
      'index3d': index3d,
    };
    await _writeJsonAtomic(File(p.join(dir.path, 'meta.json')), payload);
  }

  Map<String, dynamic> _build2DIndex(Pose2DSequence seq, VideoMeta? meta) {
    final frames = seq.frames;
    final timestamps = frames.isEmpty
        ? const {'start': 0.0, 'end': 0.0}
        : {
            'start': double.parse(frames.first.t.toStringAsFixed(3)),
            'end': double.parse(frames.last.t.toStringAsFixed(3)),
          };
    return {
      'frameCount': frames.length,
      'fps': seq.fps,
      'keypointSet': 'H36M-17',
      'schemaVersion': 1,
      'timestamps': timestamps,
      if (meta != null) 'imgW': meta.width,
      if (meta != null) 'imgH': meta.height,
    };
  }

  Map<String, dynamic> _build3DIndex(Pose3DResult result) => {
        'windowCount': result.windows.length,
        'skeleton': 'H36M-17',
        'normalization': 'center-subtract, scale=min(W,H)/2',
        'schemaVersion': 1,
        'windowSize': result.windowSize,
        'stride': result.stride,
        'pelvisCentered': result.pelvisCentered,
      };

  Future<void> _writeJsonAtomic(File file, Map<String, dynamic> data) async {
    final tmp = File('${file.path}.tmp');
    await tmp.writeAsString(jsonEncode(data));
    if (await file.exists()) {
      await file.delete();
    }
    await tmp.rename(file.path);
  }

  Future<void> _deleteDir(Directory dir) async {
    if (await dir.exists()) {
      await dir.delete(recursive: true);
    }
  }
}

SessionId generateSessionId() {
  final now = DateTime.now().toUtc();
  final base = now.toIso8601String().replaceAll(':', '-').replaceAll('.', '-');
  final rand = Random().nextInt(0xFFFFFF).toRadixString(16).padLeft(6, '0');
  return '${base}_$rand';
}
