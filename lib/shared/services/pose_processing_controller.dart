// lib/shared/services/pose_processing_controller.dart
//
// Drop-in controller that:
// 1) Runs MotionBERT after live 2D recording: run3DForSession(sessionId, Size).
// 2) Restores legacy API surface so existing widgets/screens compile:
//    - ProgressStatus.determinate
//    - ProgressEvent.processed/total/allowCancel/isHidden
//    - isProcessing getter
//    - generateSessionId()
//    - VideoMeta class
//    - PoseProcessingCancelled
//    - processRecording(...) stub (throws if called; use run3DForSession instead).
//
// NOTE: This file does not implement the old video->2D extraction pipeline.
//       Your app is on Option B (live 2D streaming). Keep using run3DForSession.

import 'dart:async';
import 'dart:io';
import 'dart:ui' show Size;

import 'package:flutter/foundation.dart';
import 'package:path/path.dart' as p;

import 'storage_layout.dart';
import 'video_pose_pipeline.dart';

typedef SessionId = String;

/// Legacy-compatible status set (widgets like ProcessingBanner rely on these).
enum ProgressStatus { hidden, indeterminate, determinate, complete, error, cancelled }

/// Legacy-compatible progress payload used by ProcessingBanner, etc.
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
  final double? value;     // 0..1 for determinate
  final int? processed;    // items processed so far
  final int? total;        // total items expected
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

  @override
  String toString() => 'ProgressEvent('
      'status=$status, phase=$phase, value=$value, processed=$processed, total=$total, message=$message)';
}

/// Legacy helper your UI constructs alongside session metadata.
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

  @override
  String toString() =>
      'VideoMeta(sessionId=$sessionId, fps=$fps, WxH=${width}x$height, exercise=$exercise, '
      'start=$startTime, end=$endTime, platform=$platform, app=$appVersion)';
}

/// Exception type used by older catch blocks in screens.
class PoseProcessingCancelled implements Exception {
  const PoseProcessingCancelled([this.message]);
  final String? message;
  @override
  String toString() => 'PoseProcessingCancelled(${message ?? ''})';
}

/// Main controller used by UI. Keeps legacy fields/methods but focuses on 3D run.
class PoseProcessingController extends ChangeNotifier {
  PoseProcessingController({VideoPosePipeline? pipeline})
      : _pipeline = pipeline ?? VideoPosePipeline();

  final VideoPosePipeline _pipeline;

  final _progressCtrl = StreamController<ProgressEvent>.broadcast();
  Stream<ProgressEvent> get progressStream => _progressCtrl.stream;

  bool _busy = false;

  /// Legacy alias used by screens/widgets.
  bool get isProcessing => _busy;

  /// New API for Option B: run only the MotionBERT step after live 2D is saved.
  ///
  /// Returns the generated out_3d.json on success; null on failure.
  Future<File?> run3DForSession(
    SessionId sessionId,
    Size frameSize, {
    File? videoFile,
  }) async {
    if (_busy) {
      debugPrint('[PoseProcessingController] Busy; run3DForSession ignored.');
      return null;
    }
    _busy = true;
    _progressCtrl.add(ProgressEvent.indeterminate(phase: 'Preparing pipeline'));

    try {
      debugPrint('[PoseProcessingController] run3DForSession session=$sessionId frameSize=$frameSize');

      File? inputVideo = videoFile;
      if (inputVideo == null) {
        final Directory dir = await StorageLayout.sessionDir(sessionId);
        final File candidate = File(p.join(dir.path, 'video.mp4'));
        if (!await candidate.exists()) {
          throw StateError('Session video not found. Provide videoFile when calling run3DForSession.');
        }
        inputVideo = candidate;
      }

      _progressCtrl.add(ProgressEvent.indeterminate(phase: 'Running offline 2D'));

      final summary = await _pipeline.run(
        video: inputVideo,
        sessionId: sessionId,
      );

      _progressCtrl.add(ProgressEvent.complete(phase: 'Pipeline complete'));
      debugPrint('[PoseProcessingController] Pipeline wrote: ${summary.out3d.path}');
      return summary.out3d;
    } on FileSystemException catch (e, st) {
      debugPrint('[PoseProcessingController][FS-ERROR] ${e.message}\n$st');
      _progressCtrl.add(ProgressEvent.error(e.message));
      return null;
    } on PoseProcessingCancelled {
      _progressCtrl.add(ProgressEvent.cancelled());
      return null;
    } catch (e, st) {
      debugPrint('[PoseProcessingController][ERROR] $e\n$st');
      _progressCtrl.add(ProgressEvent.error(e.toString()));
      return null;
    } finally {
      _busy = false;
    }
  }

  /// Legacy pipeline entry point. Kept only so existing code compiles.
  /// It *does not* implement the old video->2D->3D steps anymore.
  /// If your UI still calls this, adapt it to use run3DForSession instead.
  Future<void> processRecording({
    required File videoFile,
    required VideoMeta meta,
  }) async {
    // Keep the method so older screens compile, but fail fast if used.
    // Replace those call sites to use run3DForSession(meta.sessionId, Size(meta.width.toDouble(), meta.height.toDouble())).
    throw UnimplementedError(
      'processRecording() is no longer supported in Option B. '
      'Use run3DForSession(sessionId, Size(w,h)) after live 2D streaming.',
    );
  }

  /// Optional cancel hook (not wired for ORT yet; kept for legacy UI).
  void cancel() {
    debugPrint('[PoseProcessingController] cancel() requested (no-op)');
    _progressCtrl.add(ProgressEvent.cancelled());
  }

  @override
  void dispose() {
    _progressCtrl.close();
    super.dispose();
  }

  // ───────────────────────────── helpers ─────────────────────────────

}

/// Legacy helper used by some screens to generate unique session ids.
SessionId generateSessionId() {
  final now = DateTime.now().toUtc();
  final base = now.toIso8601String().replaceAll(':', '-').replaceAll('.', '-');
  final rand = Random().nextInt(0xFFFFFF).toRadixString(16).padLeft(6, '0');
  return '${base}_$rand';
}
