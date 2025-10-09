// lib/shared/services/video_sampler.dart
import 'dart:io';

import 'package:ffmpeg_kit_flutter_new/ffmpeg_kit.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

class VideoFrameBatch {
  VideoFrameBatch({required this.directory, required this.files});

  final Directory directory;
  final List<File> files;

  Future<void> cleanup() async {
    if (await directory.exists()) {
      await directory.delete(recursive: true);
    }
  }
}

class VideoSampler {
  static Future<List<File>> extract10FpsJpgs(File src) async {
    final tmp = await getTemporaryDirectory();
    final outDir = Directory(p.join(tmp.path, 'fp_frames_${DateTime.now().millisecondsSinceEpoch}'));
    await outDir.create(recursive: true);

    final pattern = p.join(outDir.path, 'frame_%05d.jpg');

    final cmd = [
      '-y', '-i', src.path,
      '-vf',
      "fps=10,"
      "scale=640:640:force_original_aspect_ratio=decrease:force_divisible_by=2:flags=lanczos,"
      "pad=640:640:(ow-iw)/2:(oh-ih)/2",
      '-qscale:v', '2',
      pattern
    ].join(' ');


    final sess = await FFmpegKit.execute(cmd);
    final rc = await sess.getReturnCode();
    if (rc?.isValueSuccess() != true) {
      throw Exception('FFmpeg failed: $rc');
    }

    final files = outDir
        .listSync()
        .whereType<File>()
        .where((f) => f.path.toLowerCase().endsWith('.jpg'))
        .toList()
      ..sort((a, b) => a.path.compareTo(b.path));

    return files;
  }

  static Future<VideoFrameBatch> extractFramesAtFps(
    File src,
    double fps, {
    int? maxFrames,
  }) async {
    final tmp = await getTemporaryDirectory();
    final outDir = Directory(
      p.join(tmp.path, 'fp_custom_${DateTime.now().millisecondsSinceEpoch}_${fps.toStringAsFixed(2)}'),
    );
    await outDir.create(recursive: true);

    final pattern = p.join(outDir.path, 'frame_%05d.jpg');
    final clampedFps = fps.isFinite && fps > 0 ? fps : 1.0;
    final args = <String>['-y', '-i', src.path, '-vf', 'fps=${clampedFps.toStringAsFixed(3)}'];
    if (maxFrames != null && maxFrames > 0) {
      args.addAll(['-frames:v', maxFrames.toString()]);
    }
    args.add(pattern);

    final sess = await FFmpegKit.execute(args.join(' '));
    final rc = await sess.getReturnCode();
    if (rc?.isValueSuccess() != true) {
      await outDir.delete(recursive: true);
      throw Exception('FFmpeg failed: $rc');
    }

    final files = outDir
        .listSync()
        .whereType<File>()
        .where((f) => f.path.toLowerCase().endsWith('.jpg'))
        .toList()
      ..sort((a, b) => a.path.compareTo(b.path));

    return VideoFrameBatch(directory: outDir, files: files);
  }
}
