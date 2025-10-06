import 'dart:io';

import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

/// Centralizes session file layout so every component shares the same paths.
class StorageLayout {
  StorageLayout._();

  static Future<Directory> _baseDir() async {
    final docs = await getApplicationDocumentsDirectory();
    final dir = Directory(p.join(docs.path, 'FitPerfect'));
    if (!await dir.exists()) {
      await dir.create(recursive: true);
    }
    return dir;
  }

  static Future<Directory> sessionDir(String sessionId) async {
    final base = await _baseDir();
    final dir = Directory(p.join(base.path, sessionId));
    if (!await dir.exists()) {
      await dir.create(recursive: true);
    }
    return dir;
  }

  static Future<Directory> logsDir(String sessionId) async {
    final dir = Directory(p.join((await sessionDir(sessionId)).path, 'logs'));
    if (!await dir.exists()) {
      await dir.create(recursive: true);
    }
    return dir;
  }

  static Future<Directory> cfgDir(String sessionId) async {
    final dir = Directory(p.join((await sessionDir(sessionId)).path, 'cfg'));
    if (!await dir.exists()) {
      await dir.create(recursive: true);
    }
    return dir;
  }

  static Future<File> out2dFile(String sessionId) async {
    final dir = await sessionDir(sessionId);
    return File(p.join(dir.path, 'out_2d.jsonl'));
  }

  static Future<File> cocoShadowFile(String sessionId) async {
    final dir = await sessionDir(sessionId);
    return File(p.join(dir.path, 'coco_2d.jsonl'));
  }

  static Future<File> out2dIndexFile(String sessionId) async {
    final dir = await sessionDir(sessionId);
    return File(p.join(dir.path, 'out_2d_index.json'));
  }

  static Future<File> out3dFile(String sessionId) async {
    final dir = await sessionDir(sessionId);
    return File(p.join(dir.path, 'out_3d.json'));
  }

  static Future<File> out3dIndexFile(String sessionId) async {
    final dir = await sessionDir(sessionId);
    return File(p.join(dir.path, 'out_3d_index.json'));
  }

  static Future<File> metaFile(String sessionId) async {
    final dir = await sessionDir(sessionId);
    return File(p.join(dir.path, 'meta.json'));
  }

  static Future<File> yoloDecodeLog(String sessionId) async {
    final dir = await logsDir(sessionId);
    return File(p.join(dir.path, 'yolo_decode.txt'));
  }

  static Future<File> timingsLog(String sessionId) async {
    final dir = await logsDir(sessionId);
    return File(p.join(dir.path, 'timings.json'));
  }

  static Future<File> pipelineLog(String sessionId) async {
    final dir = await logsDir(sessionId);
    return File(p.join(dir.path, 'pipeline.txt'));
  }

  static Future<File> legacyFramesJsonl(String sessionId) async {
    final base = await _baseDir();
    final legacy = Directory(p.join(base.path, 'poses', sessionId, '2d'));
    return File(p.join(legacy.path, 'frames.jsonl'));
  }

  static Future<File> localVideoFile(String exercise, DateTime timestamp) async {
    final base = await _baseDir();
    final name = 'local_${exercise}_${timestamp.toIso8601String().replaceAll(':', '-')}.mp4';
    return File(p.join(base.path, name));
  }
}
