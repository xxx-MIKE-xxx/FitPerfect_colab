// lib/shared/services/pose_processing_controller.dart
import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:isolate';
import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

/// Identifies the current lifecycle state of the processing controller.
enum ProcessingState { idle, streaming, stopping, running3D, done, error }

/// Session metadata returned when starting a live capture run.
class SessionInfo {
  SessionInfo({
    required this.sessionId,
    required this.dir,
    required this.jsonlPath,
  });

  final String sessionId;
  final Directory dir;
  final String jsonlPath;
}

/// Final result produced after running MotionBERT on a captured session.
class ProcessingResult {
  ProcessingResult({
    required this.jsonl2dPath,
    required this.out3dPath,
    this.reportPaths = const <String>[],
  });

  final String jsonl2dPath;
  final String out3dPath;
  final List<String> reportPaths;
}

/// Controller that orchestrates the live 2D capture session and the 3D post step.
class PoseProcessingController {
  PoseProcessingController({
    required LivePoseEngine engine,
    MotionBertRunner? motionBertRunner,
  })  : _engine = engine,
        _motionBertRunner = motionBertRunner ?? const MotionBertRunner();

  final LivePoseEngine _engine;
  final MotionBertRunner _motionBertRunner;

  final _stateCtl = StreamController<ProcessingState>.broadcast();
  final _errCtl = StreamController<Object>.broadcast();

  StreamSubscription<Object>? _engineErrSub;
  ProcessingState _state = ProcessingState.idle;
  SessionInfo? _activeSession;
  IOSink? _jsonlSink;
  bool _disposed = false;

  Stream<ProcessingState> get state => _stateCtl.stream;

  Stream<Object> get errors => _errCtl.stream;

  ProcessingState get currentState => _state;

  /// Starts the live pose engine and opens the JSONL writer for the session.
  Future<SessionInfo> startLive({
    int frameStride = 3,
    bool writeToDocuments = true,
  }) async {
    _ensureNotDisposed();
    if (_state != ProcessingState.idle &&
        _state != ProcessingState.done &&
        _state != ProcessingState.error) {
      throw StateError('Cannot start a new session while $_state');
    }

    await _teardownSession();

    final sessionId = generateSessionId();
    final baseDir =
        writeToDocuments ? await _documentsDir() : await _temporaryDir();
    final sessionDir = Directory(p.join(baseDir.path, 'FitPerfect', sessionId));
    await sessionDir.create(recursive: true);

    final jsonlFile = File(p.join(sessionDir.path, 'coco_2d.jsonl'));
    final sink = jsonlFile.openWrite(mode: FileMode.append);

    final session = SessionInfo(
      sessionId: sessionId,
      dir: sessionDir,
      jsonlPath: jsonlFile.path,
    );

    _jsonlSink = sink;
    _activeSession = session;

    _setState(ProcessingState.streaming);

    await _engineErrSub?.cancel();
    _engineErrSub = _engine.errors.listen(
      (error) => _errCtl.add(error),
      onError: (Object error, StackTrace stackTrace) {
        _errCtl.add(error);
        _setState(ProcessingState.error);
      },
    );

    try {
      await _engine.start(
        session: session,
        jsonlSink: sink,
        frameStride: frameStride,
      );
    } catch (error, stackTrace) {
      await _handleStartFailure(error, stackTrace);
      rethrow;
    }

    return session;
  }

  /// Stops the live engine, finalizes 2D output, and runs MotionBERT.
  Future<ProcessingResult> stopAndRun3D() async {
    _ensureNotDisposed();
    final session = _activeSession;
    final sink = _jsonlSink;
    if (session == null || sink == null) {
      throw StateError('No active live session to stop');
    }

    _setState(ProcessingState.stopping);

    try {
      await _engine.stop();
    } catch (error, stackTrace) {
      _errCtl.add(error);
      if (kDebugMode) {
        debugPrint('LivePoseEngine.stop error: $error\n$stackTrace');
      }
      _setState(ProcessingState.error);
      rethrow;
    }

    await sink.flush();
    await sink.close();
    _jsonlSink = null;

    final jsonlFile = File(session.jsonlPath);
    if (!await jsonlFile.exists()) {
      _setState(ProcessingState.error);
      throw StateError('Expected JSONL at ${session.jsonlPath} after stop');
    }
    final stat = await jsonlFile.stat();
    if (stat.size <= 0) {
      _setState(ProcessingState.error);
      throw StateError('JSONL file at ${session.jsonlPath} is empty');
    }

    _setState(ProcessingState.running3D);

    ProcessingResult result;
    try {
      result = await _motionBertRunner.run(
        jsonl2dPath: session.jsonlPath,
        sessionDir: session.dir,
        providers: _defaultProviders(),
      );
    } catch (error, stackTrace) {
      _errCtl.add(error);
      if (kDebugMode) {
        debugPrint('MotionBertRunner error: $error\n$stackTrace');
      }
      _setState(ProcessingState.error);
      rethrow;
    }

    _setState(ProcessingState.done);
    _activeSession = null;
    return result;
  }

  /// Disposes resources and stops any ongoing session best-effort.
  Future<void> dispose() async {
    if (_disposed) return;
    _disposed = true;

    await _engineErrSub?.cancel();
    _engineErrSub = null;

    if (_state == ProcessingState.streaming) {
      try {
        await _engine.stop();
      } catch (error, stackTrace) {
        if (kDebugMode) {
          debugPrint('Error stopping engine during dispose: $error\n$stackTrace');
        }
        _errCtl.add(error);
      }
    }

    final sink = _jsonlSink;
    if (sink != null) {
      await sink.flush();
      await sink.close();
      _jsonlSink = null;
    }

    try {
      await _engine.dispose();
    } catch (error, stackTrace) {
      if (kDebugMode) {
        debugPrint('Error disposing engine: $error\n$stackTrace');
      }
    }

    await _stateCtl.close();
    await _errCtl.close();
    _activeSession = null;
    _setState(ProcessingState.idle);
  }

  Future<Directory> _documentsDir() async {
    final dir = await getApplicationDocumentsDirectory();
    return dir;
  }

  Future<Directory> _temporaryDir() async {
    final dir = await getTemporaryDirectory();
    return dir;
  }

  void _setState(ProcessingState next) {
    if (_state == next) return;
    _state = next;
    if (!_stateCtl.isClosed) {
      _stateCtl.add(next);
    }
  }

  Future<void> _handleStartFailure(Object error, StackTrace stackTrace) async {
    _errCtl.add(error);
    if (kDebugMode) {
      debugPrint('LivePoseEngine.start error: $error\n$stackTrace');
    }
    await _jsonlSink?.flush();
    await _jsonlSink?.close();
    _jsonlSink = null;
    if (_activeSession != null) {
      try {
        if (await _activeSession!.dir.exists()) {
          await _activeSession!.dir.delete(recursive: true);
        }
      } catch (_) {}
    }
    _activeSession = null;
    _setState(ProcessingState.error);
  }

  Future<void> _teardownSession() async {
    if (_jsonlSink != null) {
      await _jsonlSink!.flush();
      await _jsonlSink!.close();
    }
    _jsonlSink = null;
    _activeSession = null;
  }

  void _ensureNotDisposed() {
    if (_disposed) {
      throw StateError('PoseProcessingController has been disposed');
    }
  }

  List<String> _defaultProviders() {
    if (kIsWeb) return const ['cpu'];
    if (Platform.isAndroid) {
      return const ['nnapi', 'xnnpack', 'cpu'];
    }
    if (Platform.isIOS) {
      return const ['coreml', 'cpu'];
    }
    return const ['cpu'];
  }
}

/// Helper that executes the MotionBERT model (placeholder implementation).
class MotionBertRunner {
  const MotionBertRunner();

  Future<ProcessingResult> run({
    required String jsonl2dPath,
    required Directory sessionDir,
    required List<String> providers,
  }) async {
    return Isolate.run<ProcessingResult>(() async {
      final jsonlFile = File(jsonl2dPath);
      if (!await jsonlFile.exists()) {
        throw StateError('Missing 2D JSONL at $jsonl2dPath');
      }

      final lines = await jsonlFile.readAsLines();
      final outPath = p.join(sessionDir.path, 'out_3d.json');

      final payload = <String, Object?>{
        'generatedAt': DateTime.now().toUtc().toIso8601String(),
        'providers': providers,
        'frameCount': lines.length,
        'note': 'Placeholder MotionBERT output',
      };

      final outFile = File(outPath);
      await outFile.create(recursive: true);
      await outFile.writeAsString('${jsonEncode(payload)}\n');

      return ProcessingResult(
        jsonl2dPath: jsonl2dPath,
        out3dPath: outFile.path,
        reportPaths: const <String>[],
      );
    });
  }
}

/// Minimal interface required from the live engine implementation.
abstract class LivePoseEngine {
  Stream<Object> get errors;

  Future<void> start({
    required SessionInfo session,
    required IOSink jsonlSink,
    required int frameStride,
  });

  Future<void> stop();

  Future<void> dispose();
}

String generateSessionId() {
  final now = DateTime.now().toUtc();
  final base = now.toIso8601String().replaceAll(':', '-').replaceAll('.', '-');
  final rand = Random().nextInt(0xFFFFFF).toRadixString(16).padLeft(6, '0');
  return '${base}_$rand';
}
