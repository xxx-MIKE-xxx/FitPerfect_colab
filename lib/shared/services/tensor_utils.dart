// lib/shared/services/tensor_utils.dart
import 'package:image/image.dart' as img;

// Some image v4 Pixel fields are `num`, so cast to int safely.
int _asInt(num v) => v is int ? v : v.toInt();

/// Works with both image v3 (int pixels) and v4 (Pixel objects).
/// For the rare int path we assume 0xAARRGGBB layout.
int getRed(dynamic c)   => c is img.Pixel ? _asInt(c.r) : (c is int ? (c >> 16) & 0xFF : 0);
int getGreen(dynamic c) => c is img.Pixel ? _asInt(c.g) : (c is int ? (c >> 8)  & 0xFF : 0);
int getBlue(dynamic c)  => c is img.Pixel ? _asInt(c.b) : (c is int ?  c        & 0xFF : 0);
int getAlpha(dynamic c) => c is img.Pixel ? _asInt(c.a) : (c is int ? (c >> 24) & 0xFF : 255);
