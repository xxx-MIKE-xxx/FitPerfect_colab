// lib/shared/services/summary_repository.dart
import 'dart:convert';
import 'dart:io';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

import 'api_client.dart';

const bool _enableSummaryRepositoryLogs = false;

void _summaryRepoLog(String message) {
  if (!_enableSummaryRepositoryLogs) return;
  debugPrint(message);
}

class SummaryRepository {
  /// Order:
  /// 1) Try backend summary (when s3Key provided)
  /// 2) Derive from local report (if present)
  /// 3) Load fixture from assets/fixtures/summary_<exerciseId>.json
  static Future<Map<String, dynamic>> load({
    String? s3Key,
    String? exerciseId,
    Map<String, dynamic>? localReport,
  }) async {
    // 1) Server, if we have a key
    if (s3Key != null) {
      try {
        return await ApiClient.fetchSummary(s3Key);
      } catch (e) {
        _summaryRepoLog('[SummaryRepository] server summary failed: $e');
      }
    }

    // 2) Derive from local report (same shape as the server!)
    if (localReport != null) {
      return _deriveFromLocalReport(localReport, exerciseId);
    }

    // 3) Fixture
    if (exerciseId != null) {
      try {
        final str =
            await rootBundle.loadString('assets/fixtures/summary_$exerciseId.json');
        return jsonDecode(str) as Map<String, dynamic>;
      } catch (e) {
        _summaryRepoLog('[SummaryRepository] missing fixture for $exerciseId: $e');
      }
    }

    // Final fallback: empty-looking summary
    return {
      'version': 'summary-v1',
      'exerciseId': exerciseId ?? 'unknown',
      'media': { 's3Key': s3Key, 'fps': 30 },
      'score': 80,
      'mistakes': <dynamic>[],
      'generatedAt': DateTime.now().toUtc().toIso8601String(),
    };
  }

  // Turn your current CSV-like report into the same "summary-v1" format.
  static Map<String, dynamic> _deriveFromLocalReport(
    Map<String, dynamic> report,
    String? exerciseId,
  ) {
    int score = 100;
    final mistakes = <Map<String, dynamic>>[];
    final rows = (report['data'] ?? []) as List<dynamic>;

    List<String> _asStringList(dynamic raw) {
      if (raw == null) return const [];
      if (raw is String) {
        try { return List<String>.from(jsonDecode(raw.replaceAll("'", '"'))); }
        catch (_) { return const []; }
      }
      if (raw is List) return raw.map((e) => e.toString()).toList();
      return const [];
    }

    List<List<int>> _asRanges(dynamic raw) {
      if (raw == null) return const [];
      try {
        final List list =
            raw is String ? jsonDecode(raw.replaceAll("'", '"')) : (raw as List);
        return list.map<List<int>>((e) => List<int>.from(e as List)).toList();
      } catch (_) {
        return const [];
      }
    }

    for (final row in rows) {
      final r = row as Map<String, dynamic>;

      // depth
      final depthSev = (r['depth_severity'] as String?) ?? 'none';
      if (depthSev != 'none') {
        if (depthSev == 'mild') score -= 5;
        if (depthSev == 'severe') score -= 12;

        final angle = (r['depth_angle_deg'] as num?)?.toDouble();
        mistakes.add({
          'code': 'depth_angle_low',
          'title': 'Depth angle',
          'severity': depthSev,
          'detail': angle == null ? null : '${angle.toStringAsFixed(1)}° (target ≥ 85°)',
          'metrics': { if (angle != null) 'angle_deg': angle },
          'ranges': [
            { 'startFrame': (r['depth_frame'] as num?)?.toInt() ?? 0,
              'endFrame': (r['depth_frame'] as num?)?.toInt() ?? 0 }
          ],
        });
      }

      // hip
      final hipSev = _asStringList(r['hip_severity']);
      final hipVal = (r['hip_value'] is String)
          ? (jsonDecode((r['hip_value'] as String).replaceAll("'", '"')) as List)
              .map((e) => (e as num).toDouble())
              .toList()
          : (r['hip_value'] as List<dynamic>?)
                  ?.map((e) => (e as num).toDouble())
                  .toList() ??
              const <double>[];
      final hipFrames = _asRanges(r['hip_frames']);

      for (var i = 0; i < hipSev.length; i++) {
        final sev = hipSev[i];
        if (sev == 'none') continue;
        if (sev == 'mild') score -= 3;
        if (sev == 'severe') score -= 8;

        final tilt = i < hipVal.length ? hipVal[i] : null;
        final range = i < hipFrames.length ? hipFrames[i] : [0, 0];

        mistakes.add({
          'code': tilt == null
              ? 'hip_shift'
              : (tilt >= 0 ? 'hip_shift_right' : 'hip_shift_left'),
          'title': 'Hip shift',
          'severity': sev,
          'detail':
              tilt == null ? null : '${tilt.abs().toStringAsFixed(2)}° ${tilt >= 0 ? 'right' : 'left'} tilt',
          'metrics': { if (tilt != null) 'tilt_deg': tilt },
          'ranges': [
            { 'startFrame': range[0], 'endFrame': range[1] }
          ],
        });
      }
    }

    score = score.clamp(0, 100);

    return {
      'version': 'summary-v1',
      'exerciseId': exerciseId ?? 'unknown',
      'media': { 's3Key': null, 'fps': 30 },
      'score': score,
      'mistakes': mistakes,
      'generatedAt': DateTime.now().toUtc().toIso8601String(),
    };
  }
}
