import 'dart:io';

import 'package:flutter/foundation.dart';       // ← for debugPrint
import 'package:flutter/services.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

class VideoTranscoder {
  static const _channel = MethodChannel('video_transcoder');

  /// Returns a new MP4 limited to 10 fps; keeps the original untouched.
  static Future<File> to10Fps(File src) async {
    debugPrint('[VideoTranscoder] Dart → native for ${src.path}');

    final tmpDir  = await getTemporaryDirectory();
    final dstPath = p.join(
      tmpDir.path,
      '${p.basenameWithoutExtension(src.path)}_10fps.mp4',
    );

    // Native side gives us the path of the transcoded file
    final String? outPath = await _channel.invokeMethod<String>('to10fps', {
      'src': src.path,
      'dst': dstPath,
    });

    if (outPath == null) {
      throw PlatformException(
        code: 'NATIVE_FAILED',
        message: 'Native transcoder returned null path',
      );
    }

    debugPrint('[VideoTranscoder] native returned $outPath');
    return File(outPath);
  }
}
