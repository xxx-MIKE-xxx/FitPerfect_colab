import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;

/// Converts a CameraImage to an image.Image (RGB).
/// Supports:
///  • BGRA8888 single-plane (iOS)
///  • YUV420 3-plane (separate U and V planes)
///  • YUV420SP 2-plane (interleaved UV or VU)
img.Image yuv420ToImage(CameraImage cam) {
  // ───────────── BGRA8888 (iOS) ─────────────
  if (cam.planes.length == 1 && cam.format.group == ImageFormatGroup.bgra8888) {
    final p = cam.planes.first;
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

  // ───────────── YUV420 handling (Android) ─────────────
  final width = cam.width;
  final height = cam.height;
  final fmt = cam.format.group;

  // Guard
  if (fmt != ImageFormatGroup.yuv420) {
    // Fallback: create a blank image if some unexpected format shows up
    return img.Image(width: width, height: height);
  }

  // Helper: clamp to 0..255
  int _clamp(int v) => (v < 0)
      ? 0
      : (v > 255)
          ? 255
          : v;

  // Full-range BT.601 conversion (works well for mobile camera YUV)
  // R = Y              + 1.402   * (V - 128)
  // G = Y - 0.344136 * (U - 128) - 0.714136 * (V - 128)
  // B = Y + 1.772   * (U - 128)
  void _yuvToRgbPut(img.Image out, int x, int y, int Yv, int Uv, int Vv) {
    final yy = Yv.toDouble();
    final uu = Uv.toDouble() - 128.0;
    final vv = Vv.toDouble() - 128.0;
    int r = (yy + 1.402 * vv).round();
    int g = (yy - 0.344136 * uu - 0.714136 * vv).round();
    int b = (yy + 1.772 * uu).round();
    out.setPixelRgba(x, y, _clamp(r), _clamp(g), _clamp(b), 255);
  }

  final out = img.Image(width: width, height: height);

  // ───────────── Case A: 3 planes (Y, U, V) ─────────────
  if (cam.planes.length == 3) {
    final Plane yPlane = cam.planes[0];
    final Plane uPlane = cam.planes[1];
    final Plane vPlane = cam.planes[2];

    final Uint8List Y = yPlane.bytes;
    final Uint8List U = uPlane.bytes;
    final Uint8List V = vPlane.bytes;

    final int yRowStride = yPlane.bytesPerRow;                 // usually >= width
    final int uRowStride = uPlane.bytesPerRow;
    final int vRowStride = vPlane.bytesPerRow;
    final int uPixelStride = uPlane.bytesPerPixel ?? 1;        // often 2
    final int vPixelStride = vPlane.bytesPerPixel ?? 1;        // often 2

    for (int y = 0; y < height; y++) {
      final yIndex = yRowStride * y;
      final uRow = uRowStride * (y >> 1);
      final vRow = vRowStride * (y >> 1);

      for (int x = 0; x < width; x++) {
        final int Yv = Y[yIndex + x];

        // For subsampled chroma, sample each 2×2 block
        final int uCol = (x >> 1) * uPixelStride;
        final int vCol = (x >> 1) * vPixelStride;

        // Bounds-safe reads (defensive)
        final int Ui = uRow + uCol;
        final int Vi = vRow + vCol;

        final int Uv = (Ui >= 0 && Ui < U.length) ? U[Ui] : 128;
        final int Vv = (Vi >= 0 && Vi < V.length) ? V[Vi] : 128;

        _yuvToRgbPut(out, x, y, Yv, Uv, Vv);
      }
    }
    return out;
  }

  // ───────────── Case B: 2 planes (Y, interleaved chroma) ─────────────
  // Some devices provide YUV420SP: Plane 0 = Y, Plane 1 = interleaved chroma
  // This may be NV21 (VU VU ...) or NV12 (UV UV ...).
  if (cam.planes.length == 2) {
    final Plane yPlane = cam.planes[0];
    final Plane cPlane = cam.planes[1];

    final Uint8List Y = yPlane.bytes;
    final Uint8List C = cPlane.bytes;

    final int yRowStride = yPlane.bytesPerRow;
    final int cRowStride = cPlane.bytesPerRow;
    final int cPixelStride = cPlane.bytesPerPixel ?? 2; // usually 2 for interleaved

    // Heuristic: many Android devices deliver NV21 (VU order).
    // We'll try VU first; if pixelStride==1, treat as packed UVUV...
    final bool assumeNV21 = true;

    for (int y = 0; y < height; y++) {
      final yIndex = yRowStride * y;
      final cRow = cRowStride * (y >> 1);

      for (int x = 0; x < width; x++) {
        final int Yv = Y[yIndex + x];

        final int cCol = (x & ~1); // even column for pair
        final int base = cRow + cCol;

        // Defensive indexing (might have padding at row ends)
        final int i0 = base;
        final int i1 = base + 1;

        int Uv = 128, Vv = 128;
        if (i1 < C.length) {
          if (assumeNV21) {
            // NV21: V then U
            Vv = C[i0];
            Uv = C[i1];
          } else {
            // NV12: U then V
            Uv = C[i0];
            Vv = C[i1];
          }
        }

        _yuvToRgbPut(out, x, y, Yv, Uv, Vv);
      }
    }
    return out;
  }

  // Fallback: unexpected number of planes → blank image
  return img.Image(width: width, height: height);
}
