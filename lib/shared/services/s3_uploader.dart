// lib/shared/services/s3_uploader.dart
//
// identical to the previous version, but with one extra import ↓↓↓
import 'dart:io';

import 'package:amplify_flutter/amplify_flutter.dart';
import 'package:amplify_storage_s3/amplify_storage_s3.dart';   // ← NEW

/// Tiny helper that (1) uploads a file to S3 and (2) returns a presigned URL
/// so you can open the file in a browser or hand the link to your backend.
class S3Uploader {
  /// Uploads [file] to S3 under [key] (e.g. `'private/videos/squat/…'`)
  /// and returns a *presigned* HTTPS URL.
  static Future<String> upload(File file, String key) async {
    try {
      /* ── 1. ⬆️  upload ───────────────────────────────────────────── */
      await Amplify.Storage.uploadFile(
        localFile: AWSFile.fromPath(file.path),
        path: StoragePath.fromString(key),
      ).result;

      /* ── 2. 🔗  get presigned URL ──────────────────────────────── */
      final urlResult = await Amplify.Storage.getUrl(
        path: StoragePath.fromString(key),
      ).result;

      return urlResult.url.toString();
    } on StorageException catch (e) {
      safePrint('‼︎ S3 upload failed: ${e.message}');
      rethrow;
    }
  }

  /// Convenience wrapper when you only need a presigned URL for [key].
  static Future<String> presign(String key,
      {Duration expires = const Duration(minutes: 15)}) async {
    final urlResult = await Amplify.Storage.getUrl(
      path: StoragePath.fromString(key),
      options: StorageGetUrlOptions(
        pluginOptions: S3GetUrlPluginOptions(expiresIn: expires),
      ),
    ).result;

    return urlResult.url.toString();
  }
}
