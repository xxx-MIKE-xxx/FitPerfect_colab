// lib/shared/services/api_client.dart
// ─────────────────────────────────────────────────────────
import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:amplify_auth_cognito/amplify_auth_cognito.dart';
import 'package:amplify_flutter/amplify_flutter.dart';          //  ← for FetchAuthSessionOptions
import 'package:flutter/foundation.dart';                       //  debugPrint
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';              //  offline copy

/// Central place for all calls to the Flask analyzer API.
class ApiClient {
  /// Public URL of your Flask service – change if you put it behind a proxy.
  static const _base = 'http://63.178.80.242:5001';

  /* ───────── TOKEN HANDLING ───────── */

  /// Returns a **fresh** ID-token, forcing Cognito to refresh tokens when
  /// the cached one is expired.
  static Future<String> _freshJwt() async {
    // ask Cognito for a *refreshed* session
    final session = await Amplify.Auth.fetchAuthSession(
      options: const FetchAuthSessionOptions(forceRefresh: true),
    );

    if (!session.isSignedIn) {
      throw Exception('User is not signed-in');
    }

    // the plugin-specific fields live on CognitoAuthSession
    if (session is! CognitoAuthSession) {
      throw Exception('Unexpected auth session type');
    }

    final tokens = session.userPoolTokensResult.valueOrNull;
    if (tokens == null) throw Exception('Could not obtain fresh tokens');

    return tokens.accessToken.raw;  // or .accessToken.raw – whichever the API expects
  }


  /* ───────────── MAIN API ───────────── */

  /// Polls `/api/v1/videos/<s3-key>/result` until it returns **200** with
  /// a presigned URL, then downloads & parses the JSON report.
  static Future<Map<String, dynamic>> fetchReport(
    String s3Key, {
    Duration delay = const Duration(seconds: 5),
    int max = 60,
  }) async {
    final resultUrl = Uri.parse(
      '$_base/api/v1/videos/${Uri.encodeComponent(s3Key)}/result',
    );

    for (var attempt = 0; attempt < max; attempt++) {
      debugPrint('[ApiClient] poll #${attempt + 1} → $resultUrl');

      // 1  obtain a *fresh* JWT
      late final String jwt;
      try {
        jwt = await _freshJwt();
      } on AuthException catch (e) {
        debugPrint('[ApiClient] token refresh failed → $e; retrying…');
        await Future.delayed(const Duration(seconds: 2));
        continue;                              // next loop iteration
      }

      /* ── 2. Ask Flask whether the report is ready ───────────── */
      final resp = await http.get(
        resultUrl,
        headers: {'Authorization': 'Bearer $jwt'},
      );

      debugPrint('[ApiClient]  ↳ ${resp.statusCode}  ${resp.body}');

      /* 2.a READY ─────────────────────────────────────────────── */
      if (resp.statusCode == 200) {
        final signed = jsonDecode(resp.body)['url'] as String?;
        if (signed == null) {
          throw StateError('Missing signed URL in 200 response');
        }

        debugPrint('[ApiClient] presigned URL received – downloading…');

        /* ── 3. Download the report itself ──────────────────── */
        final reportResp = await http.get(Uri.parse(signed));
        debugPrint('[ApiClient]      download '
            '${reportResp.statusCode}  '
            '${reportResp.contentLength ?? reportResp.bodyBytes.length} bytes');

        if (reportResp.statusCode == 200) {
          final jsonStr = utf8.decode(reportResp.bodyBytes);
          final decoded = jsonDecode(jsonStr);

          /* ── 4. Persist local copy (offline review) ───────── */
          try {
            final dir      = await getApplicationDocumentsDirectory();
            final fileName = s3Key.replaceAll('/', '_') + '.json';
            final file     = File('${dir.path}/$fileName');
            await file.writeAsString(jsonStr, flush: true);
            debugPrint('[ApiClient] saved to ${file.path}');
          } catch (e) {
            debugPrint('[ApiClient] could not save report locally: $e');
          }

          return decoded is List ? {'data': decoded} : decoded;
        }

        throw HttpException(
          'Could not download report (HTTP ${reportResp.statusCode})',
          uri: Uri.parse(signed),
        );
      }

      /* 2.b STILL PROCESSING ─────────────────────────────────── */
      if (resp.statusCode == 202 || resp.statusCode == 404) {
        await Future.delayed(delay);               // wait & retry
        continue;                                  // next loop iteration
      }

      /* 2.c REAL ERROR ───────────────────────────────────────── */
      throw HttpException(
        'Backend error (HTTP ${resp.statusCode}): ${resp.body}',
        uri: resultUrl,
      );
    }

    throw TimeoutException(
      'Analysis did not finish within ${delay.inSeconds * max}s',
    );
  }
}
