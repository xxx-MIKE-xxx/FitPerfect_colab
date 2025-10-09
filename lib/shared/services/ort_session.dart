
// lib/shared/services/ort_session.dart
import 'dart:io' show Platform;
import 'dart:typed_data';
import 'package:flutter/services.dart' show rootBundle, ByteData;
import 'package:onnxruntime/onnxruntime.dart';

/// Tiny helper to load ONNX models from Flutter assets into ONNX Runtime.
class OrtManager {
  /// Back-compat convenience: load a session with a reasonable default set of EPs.
  /// (Only EPs available on the current platform are actually appended.)
  static Future<OrtSession> fromAsset(String assetPath) async {
    return fromAssetWithProviders(
      assetPath,
      providers: const ['coreml', 'nnapi', 'xnnpack', 'cpu'],
    );
  }

  /// Load an ONNX model with explicit Execution Providers and threading.
  /// Supported provider strings (case-insensitive):
  ///   - 'cpu'     (always available)
  ///   - 'xnnpack' (Android/iOS optimized CPU)
  ///   - 'nnapi'   (Android)
  ///   - 'coreml'  (iOS/macOS)
  static Future<OrtSession> fromAssetWithProviders(
    String assetPath, {
    List<String> providers = const ['cpu'],
    int intraOpNumThreads = 1,
    int interOpNumThreads = 1,
    GraphOptimizationLevel graphOptimizationLevel =
        GraphOptimizationLevel.ortEnableAll,
  }) async {
    final ByteData data = await rootBundle.load(assetPath);
    final Uint8List bytes =
        data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);

    try {
      OrtEnv.instance.init(level: OrtLoggingLevel.warning);
    } catch (_) {
      // Environment may already be initialized or the plugin may not expose init options.
    }

    final options = OrtSessionOptions();

    // Threads + graph opts
    options.setIntraOpNumThreads(intraOpNumThreads);
    options.setInterOpNumThreads(interOpNumThreads);
    options.setSessionGraphOptimizationLevel(graphOptimizationLevel);

    // Append EPs in priority order (first has highest priority)
    for (final raw in providers) {
      final p = raw.toLowerCase().trim();
      switch (p) {
        case 'xnnpack':
          options.appendXnnpackProvider();
          break;
        case 'nnapi':
          if (Platform.isAndroid) {
            // Use NNAPI defaults, you can OR flags if needed, e.g. NnapiFlags.useFp16 | NnapiFlags.cpuDisabled
            options.appendNnapiProvider(NnapiFlags.useNone);
          }
          break;
        case 'coreml':
          if (Platform.isIOS || Platform.isMacOS) {
            // Use CoreML defaults; change to CoreMLFlags.useCpuOnly if desired.
            options.appendCoreMLProvider(CoreMLFlags.useNone);
          }
          break;
        case 'cpu':
          options.appendCPUProvider(CPUFlags.useNone);
          break;
        default:
          // Unknown string -> ignore
          break;
      }
    }

    final session = OrtSession.fromBuffer(bytes, options);
    return session;
  }
}
