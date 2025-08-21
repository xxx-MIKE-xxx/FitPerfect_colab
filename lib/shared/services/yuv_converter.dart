import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;

/// Converts a CameraImage to an image.Image (RGB).
/// Works with YUV420 (preferred) and falls back to BGRA8888 on iOS if needed.
img.Image yuv420ToImage(CameraImage cam) {
  // BGRA path (iOS may deliver this if imageFormatGroup wasn't set)
  if (cam.planes.length == 1 && cam.format.group == ImageFormatGroup.bgra8888) {
    final p = cam.planes.first;
    // Plane is BGRA; convert to RGB
    final out = img.Image(width: cam.width, height: cam.height);
    final data = p.bytes;
    int si = 0;
    for (int y = 0; y < cam.height; y++) {
      for (int x = 0; x < cam.width; x++) {
        final b = data[si + 0];
        final g = data[si + 1];
        final r = data[si + 2];
        // data[si + 3] is alpha (ignored)
        out.setPixelRgba(x, y, r, g, b, 255);
        si += 4;
      }
    }
    return out;
  }

  // YUV420 path (expected)
  final Y = cam.planes[0].bytes;
  final U = cam.planes[1].bytes;
  final V = cam.planes[2].bytes;

  final width = cam.width;
  final height = cam.height;
  final out = img.Image(width: width, height: height);

  final uvRowStride = cam.planes[1].bytesPerRow;
  final uvPixelStride = cam.planes[1].bytesPerPixel ?? 1;
  final yRowStride = cam.planes[0].bytesPerRow;

  for (int y = 0; y < height; y++) {
    final pY = yRowStride * y;
    final uvRow = uvRowStride * (y >> 1);
    for (int x = 0; x < width; x++) {
      final Yv = Y[pY + x];
      final uvCol = (x >> 1) * uvPixelStride;
      final Uv = U[uvRow + uvCol];
      final Vv = V[uvRow + uvCol];

      // YUV → RGB (BT.601)
      final yy = Yv.toDouble();
      final uu = Uv.toDouble() - 128.0;
      final vv = Vv.toDouble() - 128.0;

      int r = (yy + 1.402 * vv).round();
      int g = (yy - 0.344136 * uu - 0.714136 * vv).round();
      int b = (yy + 1.772 * uu).round();

      if (r < 0) r = 0; else if (r > 255) r = 255;
      if (g < 0) g = 0; else if (g > 255) g = 255;
      if (b < 0) b = 0; else if (b > 255) b = 255;

      out.setPixelRgba(x, y, r, g, b, 255);
    }
  }
  return out;
}
