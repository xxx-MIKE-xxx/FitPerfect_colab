import 'package:flutter/foundation.dart';

typedef LogFn = void Function(String message);

class Log {
  Log._();

  static DateTime _lastCoreml = DateTime.fromMillisecondsSinceEpoch(0);

  static void i(String tag, String message) {
    debugPrint('[$tag] $message');
  }

  static void w(String tag, String message) {
    debugPrint('⚠️ [$tag] $message');
  }

  static void e(String tag, String message) {
    debugPrint('🛑 [$tag] $message');
  }

  static void coremlOnce(String message) {
    final now = DateTime.now();
    if (now.difference(_lastCoreml).inSeconds >= 5) {
      _lastCoreml = now;
      w('CoreML', message);
    }
  }
}
