// lib/shared/services/video_transcoder.dart
import 'dart:io';

import 'package:flutter/foundation.dart'; // for debugPrint
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:ffmpeg_kit_flutter_new/ffmpeg_kit.dart';

class VideoTranscoder {
  /// Creates a 10-fps, square-padded surrogate MP4 without touching [src].
  /// Output is centered inside 640×640, AR preserved, even dimensions enforced.
  static Future<File> to10Fps(File src) async {
    final tmpDir = await getTemporaryDirectory();
    final outPath =
        p.join(tmpDir.path, '${p.basenameWithoutExtension(src.path)}_10fps.mp4');

    // Clean previous file if exists
    final outFile = File(outPath);
    if (await outFile.exists()) {
      await outFile.delete();
    }

    // Safe portrait/landscape pipeline:
    //  - fps=10
    //  - scale to fit inside 640×640 (no dimension will exceed 640)
    //  - force_divisible_by=2 to keep even sizes
    //  - pad up to 640×640 (centered)
    const vfilter =
        'fps=10,'
        'scale=640:640:force_original_aspect_ratio=decrease:force_divisible_by=2:flags=lanczos,'
        'pad=640:640:(ow-iw)/2:(oh-ih)/2';

    final cmd = [
      '-y',
      '-i', _q(src.path),
      '-vf', vfilter,
      // Encode to a broadly compatible H.264 stream
      '-c:v', 'libx264',
      '-pix_fmt', 'yuv420p',
      '-preset', 'veryfast',
      '-crf', '22',
      // keep original audio out to save size / speed; add '-an' to drop audio
      '-an',
      '-movflags', '+faststart',
      _q(outPath),
    ].join(' ');

    debugPrint('[VideoTranscoder] running ffmpeg: $cmd');
    final session = await FFmpegKit.execute(cmd);
    final rc = await session.getReturnCode();

    if (rc?.isValueSuccess() != true) {
      final logs = await session.getAllLogsAsString();
      debugPrint('[VideoTranscoder] ffmpeg failed rc=$rc\n$logs');
      throw Exception('FFmpeg failed: $rc');
    }

    debugPrint('[VideoTranscoder] created $outPath');
    return File(outPath);
  }

  // Quote a path for shell safety
  static String _q(String s) => '"$s"';
}
