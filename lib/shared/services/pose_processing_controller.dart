// lib/shared/services/pose_processing_controller.dart
//
// Focused controller for Option B (live 2D streaming + post-stop 3D).
// - Exposes: run3DForSession(String sessionId, Size frameSize) -> Future<File?>
// - Does NOT depend on legacy ffmpeg/video pipeline.
// - Logs both primary and legacy 2D paths before invoking MotionBERT.
// - Uses MotionBertRunner (your existing runner) to execute ONNX and save out_3d.json.
//
// Typical use from ExercisePreviewScreen on Stop:
//   final out3d = await context
//       .read<PoseProcessingController>()
//       .run3DForSession(sessionId, Size(previewW, previewH));
//
// Ensure the provider for PoseProcessingController is in widget scope.

import 'dart:async';
import 'dart:io';
import 'dart:ui' show Size;

import 'package:flutter/foundation.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

import 'motionbert_runner.dart';

typedef SessionId = String;

/// Minimal progress status for the UI (optional).
enum ProgressStatus { hidden, running, complete, error }

/// Simple progress event (optional).
class ProgressEvent {
  ProgressEvent(this.status, {this.phase, this.value, this.message});

  final ProgressStatus status;
  final String? phase;
  final double? value; // 0..1
  final String? message;

  @override
  String toString() =>
      'ProgressEvent(status=$status, phase=$phase, value=$value, message=$message)';
}

/// Focused controller to run MotionBERT once 2D data exists on disk.
class PoseProcessingController extends ChangeNotifier {
  PoseProcessingController({MotionBertRunner? runner})
      : _runner = runner ?? MotionBertRunner();

  final MotionBertRunner _runner;

  final _progressCtrl = StreamController<ProgressEvent>.broadcast();
  Stream<ProgressEvent> get progressStream => _progressCtrl.stream;

  bool _busy = false;
  bool get isBusy => _busy;

  /// Run MotionBERT for a completed session.
  ///
  /// Responsibilities here:
  ///  - compute expected 2D paths (primary + legacy) and LOG them
  ///  - delegate execution to MotionBertRunner (which itself tries both paths)
  ///  - return the out_3d.json File on success, or null on failure
  ///
  /// NOTE: This method requires that the live 2D writer has already closed
  ///       its JSONL file (coco_2d.jsonl) when `stop()` was called in the
  ///       live engine. Otherwise MB will not find the 2D input.
  Future<File?> run3DForSession(SessionId sessionId, Size frameSize) async {
    if (_busy) {
      debugPrint('[PoseProcessingController] Busy; ignoring run3DForSession.');
      return null;
    }
    _busy = true;
    _progressCtrl.add(
      ProgressEvent(ProgressStatus.running, phase: 'Preparing MotionBERT'),
    );

    try {
      // Compute and log the expected 2D locations (primary + legacy).
      final paths = await _expected2DPaths(sessionId);
      debugPrint(
        '[PoseProcessingController] 2D lookup: '
        'primary="${paths.primary}", legacy="${paths.legacy}"',
      );

      // (Optional) quick existence probe to aid logs. We DO NOT fail here—
      // MotionBertRunner will try both paths and raise a precise error itself.
      final primaryExists = await File(paths.primary).exists();
      final legacyExists = await File(paths.legacy).exists();
      debugPrint(
        '[PoseProcessingController] 2D presence: '
        'primary=$primaryExists, legacy=$legacyExists',
      );

      // Delegate the heavy lifting.
      _progressCtrl.add(
        ProgressEvent(ProgressStatus.running, phase: 'Running MotionBERT'),
      );
      final out = await _runner.run(
        sessionId: sessionId,
        frameSize: frameSize,
        rootRelative: true,
      );

      debugPrint('[PoseProcessingController] MotionBERT wrote: ${out.path}');
      _progressCtrl.add(
        ProgressEvent(
          ProgressStatus.complete,
          phase: 'MotionBERT complete',
          value: 1.0,
        ),
      );
      return out;
    } on FileSystemException catch (e, st) {
      debugPrint('[PoseProcessingController][FS-ERROR] ${e.message}\n$st');
      _progressCtrl.add(
        ProgressEvent(ProgressStatus.error, message: e.message),
      );
      return null;
    } catch (e, st) {
      debugPrint('[PoseProcessingController][ERROR] $e\n$st');
      _progressCtrl.add(
        ProgressEvent(ProgressStatus.error, message: e.toString()),
      );
      return null;
    } finally {
      _busy = false;
    }
  }

  /// Optional cancel hook (no-op for now; can be made cooperative later).
  void cancel() {
    debugPrint('[PoseProcessingController] cancel() requested (no-op)');
  }

  @override
  void dispose() {
    _progressCtrl.close();
    super.dispose();
  }

  // ───────────────────────────── helpers ─────────────────────────────

  /// Compute the expected 2D file paths for a given session:
  ///  - primary: Documents/FitPerfect/<sessionId>/coco_2d.jsonl
  ///  - legacy : Documents/FitPerfect/poses/<sessionId>/2d/frames.jsonl
  Future<_Expected2DPaths> _expected2DPaths(SessionId sessionId) async {
    final docs = await getApplicationDocumentsDirectory();
    final base = p.join(docs.path, 'FitPerfect');
    final primary = p.join(base, sessionId, 'coco_2d.jsonl');
    final legacy = p.join(base, 'poses', sessionId, '2d', 'frames.jsonl');
    return _Expected2DPaths(primary: primary, legacy: legacy);
  }
}

class _Expected2DPaths {
  const _Expected2DPaths({required this.primary, required this.legacy});
  final String primary;
  final String legacy;
}
