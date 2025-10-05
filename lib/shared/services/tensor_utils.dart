
// lib/shared/services/tensor_utils.dart
import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'dart:math' as math;

/// Some image v4 Pixel fields are `num`, so cast to int safely.
int _asInt(num v) => v is int ? v : v.toInt();

/// Works with both image v3 (int pixels) and v4 (Pixel objects).
/// For the rare int path we assume 0xAARRGGBB layout.
int getRed(dynamic c)   => c is img.Pixel ? _asInt(c.r) : (c is int ? (c >> 16) & 0xFF : 0);
int getGreen(dynamic c) => c is img.Pixel ? _asInt(c.g) : (c is int ? (c >> 8)  & 0xFF : 0);
int getBlue(dynamic c)  => c is img.Pixel ? _asInt(c.b) : (c is int ?  c        & 0xFF : 0);
int getAlpha(dynamic c) => c is img.Pixel ? _asInt(c.a) : (c is int ? (c >> 24) & 0xFF : 255);

/// Convert an [img.Image] (RGB) to Float32 CHW in [0,1].
/// Returns a buffer sized 3*W*H in channel-first order (R block, then G, then B).
Float32List chwRgb255FromImage(img.Image image) {
  final int w = image.width;
  final int h = image.height;
  final int plane = w * h;
  final buf = Float32List(3 * plane);

  int pIdx = 0; // index into R/G/B planes
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      final px = image.getPixel(x, y);
      final r = (px is img.Pixel) ? (px.r / 255.0) : (getRed(px) / 255.0);
      final g = (px is img.Pixel) ? (px.g / 255.0) : (getGreen(px) / 255.0);
      final b = (px is img.Pixel) ? (px.b / 255.0) : (getBlue(px) / 255.0);

      buf[pIdx] = r;
      buf[plane + pIdx] = g;
      buf[(plane << 1) + pIdx] = b;
      pIdx++;
    }
  }
  return buf;
}

/// Convert packed RGB bytes (RGBRGB...) to Float32 CHW in [0,1].
Float32List chwRgb255FromBytes(Uint8List rgb, int width, int height) {
  final int plane = width * height;
  final buf = Float32List(3 * plane);
  int si = 0;
  for (int p = 0; p < plane; p++) {
    final r = rgb[si++] / 255.0;
    final g = rgb[si++] / 255.0;
    final b = rgb[si++] / 255.0;
    buf[p] = r;
    buf[plane + p] = g;
    buf[(plane << 1) + p] = b;
  }
  return buf;
}

/// Clamp a double to [lo, hi].
double clampDouble(double v, double lo, double hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

/// Simple argmax over a Float32List range [offset, offset+len).
int argmax(Float32List data, int offset, int len) {
  var bestIdx = 0;
  var bestVal = double.negativeInfinity;
  for (int i = 0; i < len; i++) {
    final v = data[offset + i];
    if (v > bestVal) {
      bestVal = v;
      bestIdx = i;
    }
  }
  return bestIdx;
}

/// In-place softmax over a window in [data] from [offset] to [offset+len).
void softmaxInPlace(Float32List data, int offset, int len) {
  double maxv = double.negativeInfinity;
  for (int i = 0; i < len; i++) {
    final v = data[offset + i];
    if (v > maxv) maxv = v;
  }
  double sum = 0.0;
  for (int i = 0; i < len; i++) {
    final e = (data[offset + i] - maxv);
    final ev = e == 0.0 ? 1.0 : (e > 20 ? math.exp(20) : (e < -20 ? math.exp(-20) : math.exp(e)));
    data[offset + i] = ev;
    sum += ev;
  }
  final inv = 1.0 / sum;
  for (int i = 0; i < len; i++) {
    data[offset + i] *= inv;
  }
}

/// Minimal exp to avoid depending on dart:math everywhere else.
