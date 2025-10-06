import 'dart:io';

import 'package:path_provider/path_provider.dart';

/// Utility helpers for locating FitPerfect session artifacts on device storage.
class SessionStorage {
  const SessionStorage._();

  /// Returns the existing session directory if it was previously written either
  /// to the application's documents directory or the temporary directory.
  static Future<Directory?> findExistingSessionDir(String sessionId) async {
    final docs = await getApplicationDocumentsDirectory();
    final docsDir = Directory('${docs.path}/FitPerfect/$sessionId');
    if (await docsDir.exists()) {
      return docsDir;
    }

    final temp = await getTemporaryDirectory();
    final tempDir = Directory('${temp.path}/FitPerfect/$sessionId');
    if (await tempDir.exists()) {
      return tempDir;
    }
    return null;
  }

  /// Finds an existing file for [sessionId] named [fileName] either under the
  /// documents directory or the temporary directory.
  static Future<File?> findSessionFile(
    String sessionId,
    String fileName,
  ) async {
    final existingDir = await findExistingSessionDir(sessionId);
    if (existingDir != null) {
      final candidate = File('${existingDir.path}/$fileName');
      if (await candidate.exists()) {
        return candidate;
      }
    }

    final docs = await getApplicationDocumentsDirectory();
    final docsFile = File('${docs.path}/FitPerfect/$sessionId/$fileName');
    if (await docsFile.exists()) {
      return docsFile;
    }

    final temp = await getTemporaryDirectory();
    final tempFile = File('${temp.path}/FitPerfect/$sessionId/$fileName');
    if (await tempFile.exists()) {
      return tempFile;
    }

    return null;
  }
}
