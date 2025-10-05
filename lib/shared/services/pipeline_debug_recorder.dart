import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;

class PipelineDebugRecorder {
  PipelineDebugRecorder({
    required this.enabled,
    required this.sessionId,
    required Directory sessionDir,
    String logFileName = 'debug_log.jsonl',
  })  : _sessionDir = sessionDir,
        _sink = enabled
            ? File('${sessionDir.path}/$logFileName')
                .openWrite(mode: FileMode.append)
            : null;

  final bool enabled;
  final String sessionId;
  final Directory _sessionDir;
  final IOSink? _sink;
  final Map<String, int> _stageLastFrame = <String, int>{};
  bool _closed = false;

  Directory get sessionDir => _sessionDir;

  void log(String tag, Map<String, dynamic> payload) {
    if (!enabled || _closed) return;
    final record = <String, dynamic>{'tag': tag, ...payload};
    final line = json.encode(record);
    if (kDebugMode) {
      debugPrint(line);
    } else {
      // ignore: avoid_print
      print(line);
    }
    _sink?.writeln(line);
  }

  bool shouldLog(String tag, int frameIndex,
      {int every = 30, int first = 5}) {
    if (!enabled) return false;
    if (frameIndex < first) return true;
    if (every <= 0) return true;
    final last = _stageLastFrame[tag];
    if (last == null || frameIndex - last >= every) {
      _stageLastFrame[tag] = frameIndex;
      return true;
    }
    return false;
  }

  Future<void> saveImage(String name, img.Image image,
      {int quality = 90}) async {
    if (!enabled || _closed) return;
    final File out = File('${_sessionDir.path}/$name');
    final bytes = img.encodeJpg(image, quality: quality);
    await out.writeAsBytes(bytes, flush: true);
  }

  Future<void> saveText(String name, String contents) async {
    if (!enabled || _closed) return;
    final File out = File('${_sessionDir.path}/$name');
    await out.writeAsString(contents, flush: true);
  }

  Future<void> close() async {
    if (_closed) return;
    _closed = true;
    await _sink?.flush();
    await _sink?.close();
  }
}