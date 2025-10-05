
// lib/shared/services/yuv_converter.dart
import 'dart:typed_data';
import 'dart:io' show Platform;
import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;
import 'dart:math' as math;

/// Convert a [CameraImage] to an [img.Image] in RGB (8-bit per channel).
///
/// Supported formats:
///  • iOS: BGRA8888 (1 plane)
///  • Android: YUV420 (3 planes: Y, U, V) and YUV420SP (2 planes: Y, interleaved UV or VU).
///
/// Notes
///  - Handles arbitrary rowStride/pixelStride reported by the camera plugin.
///  - Uses BT.601 full-range integer approximation for speed.
///  - For 2-plane interleaved chroma (NV12/NV21), default behavior is chosen at runtime:
///      * If [assumeNV21] is provided, that value is used.
///      * Otherwise defaults to NV21 on Android (Platform.isAndroid), NV12 elsewhere.
img.Image yuv420ToImage(
  CameraImage cam, {
  bool? assumeNV21, // nullable default (compile-time constant) -> decide at runtime
}) {
  final int width = cam.width;
  final int height = cam.height;

  // ───────────── Case A: BGRA8888 (iOS) ─────────────
  if (cam.planes.length == 1 && cam.format.group == ImageFormatGroup.bgra8888) {
    final Plane p = cam.planes.first;
    final Uint8List src = p.bytes; // BGRA order
    final out = img.Image(width: width, height: height);
    int si = 0;
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int b = src[si++];
        final int g = src[si++];
        final int r = src[si++];
        final int a = src[si++]; // ignore alpha (opaque)
        out.setPixelRgba(x, y, r, g, b, 255);
      }
    }
    return out;
  }

  // ───────────── Case B: 3 planes (Y, U, V) ─────────────
  // Fully planar YUV420 (a.k.a. I420/YV12). U/V are quarter-res (2×2 subsampling).
  if (cam.planes.length == 3) {
    final Plane yPlane = cam.planes[0];
    final Plane uPlane = cam.planes[1];
    final Plane vPlane = cam.planes[2];

    final Uint8List Y = yPlane.bytes;
    final Uint8List U = uPlane.bytes;
    final Uint8List V = vPlane.bytes;

    final int yRowStride = yPlane.bytesPerRow;
    final int uRowStride = uPlane.bytesPerRow;
    final int vRowStride = vPlane.bytesPerRow;

    final int uPixelStride = uPlane.bytesPerPixel ?? 1; // usually 1
    final int vPixelStride = vPlane.bytesPerPixel ?? 1; // usually 1

    final out = img.Image(width: width, height: height);

    for (int y = 0; y < height; y++) {
      final int yIndex = yRowStride * y;
      final int uRow = uRowStride * (y >> 1);
      final int vRow = vRowStride * (y >> 1);
      for (int x = 0; x < width; x++) {
        final int yi = yIndex + x;
        final int ui = uRow + (x >> 1) * uPixelStride;
        final int vi = vRow + (x >> 1) * vPixelStride;

        final int Yv = (yi >= 0 && yi < Y.length) ? Y[yi] : 0;
        final int Uv = (ui >= 0 && ui < U.length) ? U[ui] : 128;
        final int Vv = (vi >= 0 && vi < V.length) ? V[vi] : 128;

        _yuvToRgbPut(out, x, y, Yv, Uv, Vv);
      }
    }
    return out;
  }

  // ───────────── Case C: 2 planes (Y, interleaved chroma) ─────────────
  // Semi-planar YUV420SP (NV12 = UVUV..., NV21 = VUVU...)
  if (cam.planes.length == 2) {
    final Plane yPlane = cam.planes[0];
    final Plane cPlane = cam.planes[1];

    final Uint8List Y = yPlane.bytes;
    final Uint8List C = cPlane.bytes;

    final int yRowStride = yPlane.bytesPerRow;
    final int cRowStride = cPlane.bytesPerRow;
    final int cPixelStride = cPlane.bytesPerPixel ?? 2; // usually 2 for interleaved

    final out = img.Image(width: width, height: height);

    // Decide NV21 vs NV12 at runtime:
    //  - If assumeNV21 is provided, use it.
    //  - Else default to NV21 on Android; NV12 elsewhere.
    final bool treatAsNV21 = assumeNV21 ?? Platform.isAndroid;

    for (int y = 0; y < height; y++) {
      final int yIndex = yRowStride * y;
      final int cRow = cRowStride * (y >> 1);
      for (int x = 0; x < width; x++) {
        final int yi = yIndex + x;
        final int cIdx = cRow + (x >> 1) * cPixelStride;

        final int Yv = (yi >= 0 && yi < Y.length) ? Y[yi] : 0;

        int Uv = 128;
        int Vv = 128;
        if (cIdx + 1 < C.length) {
          if (treatAsNV21) {
            // NV21: V then U
            Vv = C[cIdx];
            Uv = C[cIdx + 1];
          } else {
            // NV12: U then V
            Uv = C[cIdx];
            Vv = C[cIdx + 1];
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

/// Integer YUV -> RGB conversion and store to [out] at (x,y).
/// Formula is BT.601-like; we clamp to 0..255.
void _yuvToRgbPut(img.Image out, int x, int y, int Yv, int Uv, int Vv) {
  // Convert to signed space
  int C = Yv - 16;
  int D = Uv - 128;
  int E = Vv - 128;

  // Fixed-point integer math (approximate):
  // R = 1.164*C + 1.596*E
  // G = 1.164*C - 0.392*D - 0.813*E
  // B = 1.164*C + 2.017*D
  int r = (1192 * C + 1634 * E) >> 10;
  int g = (1192 * C -  400 * D -  833 * E) >> 10;
  int b = (1192 * C + 2066 * D) >> 10;

  r = r.clamp(0, 255);
  g = g.clamp(0, 255);
  b = b.clamp(0, 255);

  out.setPixelRgba(x, y, r, g, b, 255);
}
