// lib/shared/services/pose_session_storage.dart
import 'dart:convert';
import 'dart:io';

import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

import 'pose_runtime.dart';

class PoseSessionWriter {
  PoseSessionWriter({Directory? baseDir}) : _overrideBaseDir = baseDir;

  final Directory? _overrideBaseDir;
  Directory? _baseDir;

  Future<Directory> _ensureBaseDir() async {
    if (_overrideBaseDir != null) {
      _baseDir ??= _overrideBaseDir!;
      return _baseDir!;
    }
    if (_baseDir != null) return _baseDir!;
    final docs = await getApplicationDocumentsDirectory();
    final dir = Directory(p.join(docs.path, 'FitPerfect', 'poses'));
    await dir.create(recursive: true);
    _baseDir = dir;
    return dir;
  }

  static String newSessionId() {
    final now = DateTime.now().toUtc();
    final iso = now.toIso8601String().replaceAll(':', '-');
    final suffix = now.microsecondsSinceEpoch.toRadixString(16);
    return '${iso}_$suffix';
  }

  Future<Directory> sessionDirectory(String sessionId) async {
    final base = await _ensureBaseDir();
    final dir = Directory(p.join(base.path, sessionId));
    await dir.create(recursive: true);
    return dir;
  }

  Future<void> writeMeta(String sessionId, Map<String, dynamic> meta) async {
    final dir = await sessionDirectory(sessionId);
    final file = File(p.join(dir.path, 'meta.json'));
    await _atomicWrite(file, const JsonEncoder.withIndent('  ').convert(meta));
  }

  Future<Map<String, dynamic>> write2DSequence(
    String sessionId,
    Pose2DSequence sequence,
  ) async {
    final dir = await sessionDirectory(sessionId);
    final twoDDir = Directory(p.join(dir.path, '2d'));
    await twoDDir.create(recursive: true);

    final framesFile = File(p.join(twoDDir.path, 'frames.jsonl'));
    final buffer = StringBuffer();
    for (final frame in sequence.frames) {
      buffer.writeln(jsonEncode(frame.toJson()));
    }
    await _atomicWrite(framesFile, buffer.toString());

    final index = sequence.toIndexJson();
    final indexFile = File(p.join(twoDDir.path, 'index.json'));
    await _atomicWrite(indexFile, const JsonEncoder.withIndent('  ').convert(index));
    return index;
  }

  Future<Map<String, dynamic>> write3DWindows(
    String sessionId,
    List<Pose3DWindow> windows,
    MotionBertModelInfo model,
  ) async {
    final dir = await sessionDirectory(sessionId);
    final threeDDir = Directory(p.join(dir.path, '3d'));
    await threeDDir.create(recursive: true);

    final windowsFile = File(p.join(threeDDir.path, 'windows.jsonl'));
    final buffer = StringBuffer();
    for (final window in windows) {
      buffer.writeln(jsonEncode(window.toJson()));
    }
    await _atomicWrite(windowsFile, buffer.toString());

    final index = {
      'windowCount': windows.length,
      'skeleton': 'H36M-17',
      'normalization': 'image_centered_minwh',
      'schemaVersion': 1,
      'model': model.toJson(),
    };
    final indexFile = File(p.join(threeDDir.path, 'index.json'));
    await _atomicWrite(indexFile, const JsonEncoder.withIndent('  ').convert(index));
    return index;
  }

  Future<void> deleteSession(String sessionId) async {
    final base = await _ensureBaseDir();
    final dir = Directory(p.join(base.path, sessionId));
    if (await dir.exists()) {
      await dir.delete(recursive: true);
    }
  }

  Future<void> _atomicWrite(File file, String contents) async {
    final tmp = File('${file.path}.tmp');
    await tmp.writeAsString(contents, flush: true);
    if (await file.exists()) {
      await file.delete();
    }
    await tmp.rename(file.path);
  }
}
