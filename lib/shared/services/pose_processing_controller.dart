// lib/shared/services/pose_processing_controller.dart
import 'dart:async';
import 'dart:io';

import 'pose_runtime.dart';
import 'pose_session_storage.dart';

enum ProgressState { hidden, indeterminate, determinate, complete, error }

class ProgressEvent {
  ProgressEvent._(this.state, this.phase, {this.value, this.processed, this.total, this.error});

  final ProgressState state;
  final String phase;
  final double? value;
  final int? processed;
  final int? total;
  final Object? error;

  factory ProgressEvent.hidden() => ProgressEvent._(ProgressState.hidden, '');

  factory ProgressEvent.indeterminate({required String phase}) =>
      ProgressEvent._(ProgressState.indeterminate, phase);

  factory ProgressEvent.determinate({
    required String phase,
    required double value,
    int? processed,
    int? total,
  }) =>
      ProgressEvent._(
        ProgressState.determinate,
        phase,
        value: value.clamp(0.0, 1.0),
        processed: processed,
        total: total,
      );

  factory ProgressEvent.complete({String phase = 'Complete'}) =>
      ProgressEvent._(ProgressState.complete, phase, value: 1.0);

  factory ProgressEvent.error({required String phase, required Object error}) =>
      ProgressEvent._(ProgressState.error, phase, error: error);
}

class PoseProcessingRequest {
  PoseProcessingRequest({
    required this.videoFile,
    required this.sessionId,
    required this.exerciseId,
    required this.startTime,
    required this.endTime,
    this.exerciseLabel,
    this.appVersion,
  });

  final File videoFile;
  final String sessionId;
  final String exerciseId;
  final DateTime startTime;
  final DateTime endTime;
  final String? exerciseLabel;
  final String? appVersion;
}

class PoseProcessingOutcome {
  PoseProcessingOutcome({
    required this.sessionId,
    required this.sequence,
    required this.summary2D,
    required this.summary3D,
    required this.meta,
    required this.sessionDirectory,
  });

  final String sessionId;
  final Pose2DSequence sequence;
  final Map<String, dynamic> summary2D;
  final Map<String, dynamic> summary3D;
  final Map<String, dynamic> meta;
  final Directory sessionDirectory;
}

class PoseProcessingCancelled implements Exception {
  PoseProcessingCancelled();

  @override
  String toString() => 'PoseProcessingCancelled';
}

class PoseProcessingController {
  PoseProcessingController({
    PosePipeline? pipeline,
    PoseSessionWriter? storage,
  })  : _pipeline = pipeline ?? PosePipeline(),
        _storage = storage ?? PoseSessionWriter(),
        progress = StreamController<ProgressEvent>.broadcast();

  final PosePipeline _pipeline;
  final PoseSessionWriter _storage;
  final StreamController<ProgressEvent> progress;

  bool _cancelled = false;
  bool _running = false;

  Future<PoseProcessingOutcome> process(PoseProcessingRequest request) async {
    if (_running) {
      throw StateError('Pose processing already running');
    }
    _running = true;
    _cancelled = false;

    try {
      progress.add(ProgressEvent.indeterminate(phase: 'Finalizing 2D…'));
      final result = await _pipeline.analyzeVideo(
        request.videoFile,
        onProgress: (phase, {progress: value, processed, total}) {
          if (_cancelled) return;
          switch (phase) {
            case PoseProcessingPhase.sampling2d:
              this.progress.add(
                    ProgressEvent.indeterminate(phase: 'Extracting 2D frames…'));
              break;
            case PoseProcessingPhase.persisting2d:
              this.progress.add(ProgressEvent.indeterminate(
                  phase: 'Preparing 2D outputs…'));
              break;
            case PoseProcessingPhase.preparing3d:
              this.progress.add(
                    ProgressEvent.indeterminate(phase: 'Preparing 3D input…'));
              break;
            case PoseProcessingPhase.estimating3d:
              final v = value ?? 0.0;
              this.progress.add(ProgressEvent.determinate(
                phase: 'Estimating 3D poses…',
                value: v,
                processed: processed,
                total: total,
              ));
              break;
            case PoseProcessingPhase.persisting3d:
              this.progress.add(ProgressEvent.indeterminate(
                  phase: 'Finishing 3D estimation…'));
              break;
          }
        },
      );

      if (_cancelled) {
        await _storage.deleteSession(request.sessionId);
        throw PoseProcessingCancelled();
      }

      progress.add(ProgressEvent.indeterminate(phase: 'Writing 2D poses…'));
      final summary2d = await _storage.write2DSequence(
        request.sessionId,
        result.sequence2d,
      );

      if (_cancelled) {
        await _storage.deleteSession(request.sessionId);
        throw PoseProcessingCancelled();
      }

      progress.add(ProgressEvent.indeterminate(phase: 'Writing 3D windows…'));
      final summary3d = await _storage.write3DWindows(
        request.sessionId,
        result.windows3d,
        result.motionBert,
      );

      if (_cancelled) {
        await _storage.deleteSession(request.sessionId);
        throw PoseProcessingCancelled();
      }

      final info = result.motionBert;
      final meta = {
        'sessionId': request.sessionId,
        'appVersion': request.appVersion ?? 'dev',
        'platform': Platform.isIOS
            ? 'ios'
            : (Platform.isAndroid ? 'android' : Platform.operatingSystem),
        'fps': result.sequence2d.fps,
        'startTime': request.startTime.toUtc().toIso8601String(),
        'endTime': request.endTime.toUtc().toIso8601String(),
        'exercise': request.exerciseId,
        'exerciseLabel': request.exerciseLabel,
        'model2D': {
          'name': 'RTM-Pose rtm-m',
          'keypoints': 'RTM→H36M17',
          'confidenceMin': null,
        },
        'model3D': {
          'name': 'MotionBERT',
          ...info.toJson(),
        },
        'schemaVersion': 1,
      };

      progress.add(ProgressEvent.indeterminate(phase: 'Finalizing session…'));
      await _storage.writeMeta(request.sessionId, meta);

      final sessionDir = await _storage.sessionDirectory(request.sessionId);

      final outcome = PoseProcessingOutcome(
        sessionId: request.sessionId,
        sequence: result.sequence2d,
        summary2D: summary2d,
        summary3D: summary3d,
        meta: meta,
        sessionDirectory: sessionDir,
      );

      progress.add(ProgressEvent.complete());
      return outcome;
    } on PoseProcessingCancelled {
      rethrow;
    } catch (e) {
      progress.add(ProgressEvent.error(phase: 'Processing failed', error: e));
      rethrow;
    } finally {
      _running = false;
      _cancelled = false;
    }
  }

  Future<void> cancel() async {
    if (!_running) return;
    _cancelled = true;
  }
}
