// lib/shared/services/ort_session.dart
import 'dart:typed_data';
import 'package:flutter/services.dart' show rootBundle, ByteData;
import 'package:onnxruntime/onnxruntime.dart';

/// Tiny helper to load ONNX models from Flutter assets into ONNX Runtime.
class OrtManager {
  /// Load an ONNX model embedded as a Flutter asset.
  static Future<OrtSession> fromAsset(String assetPath) async {
    final ByteData data = await rootBundle.load(assetPath);
    final Uint8List bytes =
        data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);

    final options = OrtSessionOptions();
    final session = OrtSession.fromBuffer(bytes, options);
    return session;
  }
}
