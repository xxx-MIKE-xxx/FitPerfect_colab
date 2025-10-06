// lib/screens/feedback_screen.dart
// ─────────────────────────────────────────────────────────
import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:ui' show Size;

import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:video_player/video_player.dart';

import 'package:fit_perfect_v2/shared/services/pose_processing_controller.dart';
import 'package:fit_perfect_v2/shared/utils/session_storage.dart';
class FeedbackScreen extends StatefulWidget {
  const FeedbackScreen({
    super.key,
    required this.report,
    this.videoPath,
    this.videoKey,
  }) : assert(videoPath != null || videoKey != null,
            'Either videoPath or videoKey must be provided');

  final String? videoPath;          // local copy
  final String? videoKey;           // S3 key (fallback)
  final Map<String, dynamic> report;

  @override
  State<FeedbackScreen> createState() => _FeedbackScreenState();
}

/* ─────────── internal helper ─────────── */

class _ErrorEvent {
  _ErrorEvent({
    required this.frame,
    required this.label,
    required this.value,
    required this.severity,
  });

  final int    frame;      // absolute video frame
  final String label;      // e.g. 'Hip shift'
  final String value;      // numeric/string representation
  final String severity;   // none | mild | severe

  @override
  String toString() => '$label: $value  ($severity)';
}

class _FeedbackScreenState extends State<FeedbackScreen> {
  // ───── 3D MotionBERT integration state ─────
  String? _sessionId;
  String? _out3dPath;
  Map<String, dynamic>? _mbSummary;
  List<String> _logTail = const [];
  bool _loading3D = false;
  bool _retrying3D = false;
  String? _mbError;

  // ‼︎  tweak if you record with a different FPS
  static const double _assumedFps = 30.0;

  late final VideoPlayerController _vc;
  late final List<_ErrorEvent>     _events;

  /* ───────── small helper that fixes the “List<dynamic> vs String” issue ─── */

  List<dynamic> _asList(dynamic raw) {
    if (raw == null) return const [];
    if (raw is List<dynamic>) return raw;

    if (raw is String) {
      try {
        return jsonDecode(raw.replaceAll("'", '"')) as List<dynamic>;
      } catch (_) {
        /* fall through */
      }
    }
    return const [];
  }

  /* ────────────────────── REPORT → EVENT LIST ────────────────────── */

  List<_ErrorEvent> _parseReport(Map<String, dynamic> json) {
    // The ApiClient wraps CSV rows in {'data':[ ... ]}
    final rows = (json['data'] ?? []) as List<dynamic>;

    List<_ErrorEvent> evts = [];

    for (final row in rows) {
      final r = row as Map<String, dynamic>;

      // ─ depth example ───────────────────────────────────────────
      if ((r['depth_severity'] as String?) != 'none') {
        evts.add(
          _ErrorEvent(
            frame   : (r['depth_frame'] as num).toInt(),
            label   : 'Depth angle',
            value   : (r['depth_angle_deg'] as num).toStringAsFixed(1) + '°',
            severity: r['depth_severity'] as String,
          ),
        );
      }

      // ─ hip example (array fields) ──────────────────────────────
      final sevList = _asList(r['hip_severity'])
          .map((e) => e.toString())
          .toList(growable: false);
      final valList = _asList(r['hip_value'])
          .map((e) => (e as num).toDouble())
          .toList(growable: false);
      final frmList = _asList(r['hip_frames'])
          .map<List<int>>((e) => List<int>.from(e as List))
          .toList(growable: false);

      for (var i = 0; i < sevList.length; i++) {
        if (sevList[i] == 'none') continue;
        final fStart = frmList[i][0];
        final fEnd   = frmList[i][1];

        for (var f = fStart; f <= fEnd; f++) {
          evts.add(
            _ErrorEvent(
              frame   : f,
              label   : 'Hip shift',
              value   : valList[i].toStringAsFixed(2),
              severity: sevList[i],
            ),
          );
        }
      }

      // ─ add more mistake types here in the same fashion ─────────
    }

    evts.sort((a, b) => a.frame.compareTo(b.frame));
    return evts;
  }

  /* ───────────────────────── VIDEO SET-UP ───────────────────────── */

  @override
  void initState() {
    super.initState();

    _events = _parseReport(widget.report);
    _hydrateReportMetadata(widget.report);

    if (widget.videoPath != null) {
      _vc = VideoPlayerController.file(File(widget.videoPath!))
        ..initialize().then((_) {
          if (!mounted) return;
          _vc
            ..setLooping(true)
            ..play();
          setState(() {});
        });
    } else {
      final url = widget.videoKey!.startsWith('private/')
          ? widget.videoKey!.replaceFirst(
              'private/',
              'https://fitperfect-exercisevideos-deved5ec-dev.s3.eu-central-1.amazonaws.com/private/',
            )
          : widget.videoKey!;
      _vc = VideoPlayerController.networkUrl(Uri.parse(url))
        ..initialize().then((_) {
          if (!mounted) return;
          _vc
            ..setLooping(true)
            ..play();
          setState(() {});
        });
    }

    _vc.addListener(() {
      if (mounted) setState(() {});
    });

    unawaited(_bootstrap3DState());
  }

  @override
  void dispose() {
    _vc.dispose();
    super.dispose();
  }

  void _hydrateReportMetadata(Map<String, dynamic> report) {
    _sessionId = _extractSessionId(report) ?? _sessionIdFromVideoKey();
    final out3d = _extractOut3dPath(report);
    if (out3d != null && out3d.isNotEmpty) {
      _out3dPath = out3d;
    }
  }

  Future<void> _bootstrap3DState() async {
    if (_out3dPath != null && _out3dPath!.isNotEmpty) {
      await _load3D(_out3dPath!);
      return;
    }

    final sid = _sessionId;
    if (sid == null || sid.isEmpty) {
      return;
    }

    final existingOut = await SessionStorage.findSessionFile(sid, 'out_3d.json');
    if (existingOut != null) {
      if (mounted) {
        setState(() => _out3dPath = existingOut.path);
      } else {
        _out3dPath = existingOut.path;
      }
      await _load3D(existingOut.path);
    } else {
      await _loadLogTail();
    }
  }

  String? _extractSessionId(Map<String, dynamic> report) {
    const keys = ['sessionId', 'session_id', 'session', 'session-id', 'sessionID'];
    String? fromRoot = _lookupString(report, keys);
    if (fromRoot != null && fromRoot.isNotEmpty) {
      return fromRoot;
    }

    final meta = report['meta'];
    if (meta is Map<String, dynamic>) {
      final fromMeta = _lookupString(meta, keys);
      if (fromMeta != null && fromMeta.isNotEmpty) {
        return fromMeta;
      }
    }

    final data = report['data'];
    if (data is List) {
      for (final row in data) {
        if (row is Map<String, dynamic>) {
          final candidate = _lookupString(row, keys);
          if (candidate != null && candidate.isNotEmpty) {
            return candidate;
          }
        }
      }
    }
    return null;
  }

  String? _extractOut3dPath(Map<String, dynamic> report) {
    const keys = ['out3dPath', 'out_3d_path', 'out_3d', 'out3d', 'out3DPath'];
    String? path = _lookupString(report, keys);
    if (path != null && path.isNotEmpty) {
      return path;
    }

    final meta = report['meta'];
    if (meta is Map<String, dynamic>) {
      final metaPath = _lookupString(meta, keys);
      if (metaPath != null && metaPath.isNotEmpty) {
        return metaPath;
      }
    }

    final out3d = report['out3d'];
    if (out3d is Map<String, dynamic>) {
      final nested = _lookupString(out3d, const ['path', 'file', 'uri']);
      if (nested != null && nested.isNotEmpty) {
        return nested;
      }
    }

    return null;
  }

  String? _lookupString(Map<String, dynamic> source, List<String> keys) {
    for (final key in keys) {
      final value = source[key];
      final str = _valueToString(value);
      if (str != null && str.isNotEmpty) {
        return str;
      }
    }
    return null;
  }

  String? _valueToString(dynamic value) {
    if (value == null) return null;
    if (value is String) {
      final trimmed = value.trim();
      return trimmed.isEmpty ? null : trimmed;
    }
    if (value is num || value is bool) {
      return value.toString();
    }
    return null;
  }

  String? _sessionIdFromVideoKey() {
    final key = widget.videoKey;
    if (key == null || key.isEmpty) {
      return null;
    }

    final normalized = key.replaceAll('\\', '/');
    final parts = normalized.split('/');
    if (parts.isEmpty) {
      return null;
    }

    final isLocal = parts.first == 'local';
    final isPrivateVideo =
        parts.length >= 4 && parts[0] == 'private' && parts[1] == 'videos';
    if (!isLocal && !isPrivateVideo) {
      return null;
    }

    String file = '';
    for (var i = parts.length - 1; i >= 0; i--) {
      if (parts[i].isNotEmpty) {
        file = parts[i];
        break;
      }
    }

    if (file.isEmpty) {
      return null;
    }

    final dot = file.lastIndexOf('.');
    final stem = dot > 0 ? file.substring(0, dot) : file;
    return stem.isNotEmpty ? stem : null;
  }

  /* ─────────────────── SHOW TECHNIQUE ANIMATION ────────────────── */

  Future<void> _showTechnique() async {
    await showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Correct Technique'),
        content: Image.asset(
          'assets/animations/squat_correct.gif',
          fit: BoxFit.contain,
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }

  /* ───────────────────────────── UI ───────────────────────────── */

  @override
  Widget build(BuildContext context) {
    final isReady   = _vc.value.isInitialized;
    final duration  = isReady ? _vc.value.duration  : Duration.zero;
    final position  = isReady ? _vc.value.position  : Duration.zero;

    final int curFrame =
        (position.inMilliseconds / 1000.0 * _assumedFps).round();

    final curEvents = _events
        .where((e) => e.frame == curFrame)
        .toList(growable: false);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Playback'),
        actions: [
          IconButton(
            tooltip: 'Show correct technique',
            icon: const Icon(Icons.fitness_center),
            onPressed: _showTechnique,
          ),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: Center(
              child: isReady
                  ? Stack(
                      alignment: Alignment.center,
                      children: [
                        AspectRatio(
                          aspectRatio: _vc.value.aspectRatio,
                          child: VideoPlayer(_vc),
                        ),
                        if (curEvents.isNotEmpty)
                          Positioned(
                            top: 16,
                            left: 16,
                            child: _buildOverlay(curEvents, context),
                          ),
                      ],
                    )
                  : const CircularProgressIndicator(),
            ),
          ),
          if (isReady)
            Container(
              color: Theme.of(context).colorScheme.surfaceVariant,
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              child: Row(
                children: [
                  IconButton(
                    iconSize: 32,
                    icon: Icon(
                      _vc.value.isPlaying
                          ? Icons.pause_circle
                          : Icons.play_circle,
                    ),
                    onPressed: () => setState(
                      () => _vc.value.isPlaying ? _vc.pause() : _vc.play(),
                    ),
                  ),
                  Expanded(
                    child: Slider(
                      value: position.inMilliseconds
                          .toDouble()
                          .clamp(0, duration.inMilliseconds.toDouble()),
                      max: duration.inMilliseconds
                          .toDouble()
                          .clamp(1, double.infinity),
                      onChanged: (value) => _vc.seekTo(
                        Duration(milliseconds: value.toInt()),
                      ),
                    ),
                  ),
                  Text(
                    '${_fmt(position)} / ${_fmt(duration)}',
                    style: Theme.of(context).textTheme.labelSmall,
                  ),
                ],
              ),
            ),
          const SizedBox(height: 4),
          // 3D section
          _build3DSection(context),
        ],
      ),
      // New: go to Feedback Summary
      floatingActionButton: FloatingActionButton.extended(
        icon: const Icon(Icons.summarize),
        label: const Text('View summary'),
        onPressed: () {
          // Try to infer exerciseId from the key shape:
          // private/videos/<exerciseId>/timestamp.mp4
          // or local/<exerciseId>/timestamp.mp4 (bypass path)
          String exerciseId = 'unknown';
          final key = widget.videoKey ?? '';
          final parts = key.split('/');
          if (parts.length >= 3 && parts[0] == 'private' && parts[1] == 'videos') {
            exerciseId = parts[2];
          } else if (parts.length >= 2 && parts[0] == 'local') {
            exerciseId = parts[1];
          }

            context.push('/feedback-summary', extra: {
            'exerciseId': exerciseId,
            's3Key'     : widget.videoKey,
            'report'    : widget.report, // lets SummaryRepository derive/fixture if server off
          });
        },
      ),
    );
  }

  Future<void> _loadLogTail() async {
    final sid = _sessionId;
    if (sid == null) return;
    try {
      final docs = await getApplicationDocumentsDirectory();
      final dir = Directory('${docs.path}/FitPerfect/$sid');
      final file = File('${dir.path}/coco_2d.jsonl');
      if (!file.existsSync()) {
        setState(() => _logTail = const ['(no coco_2d.jsonl found yet)']);
        return;
      }
      final lines = await file.readAsLines();
      final take = lines.length >= 10 ? lines.sublist(lines.length - 10) : lines;
      if (!mounted) return;
      setState(() => _logTail = take);
    } catch (e) {
      if (!mounted) return;
      setState(() => _logTail = ['(could not read logs: $e)']);
    }
  }

  Future<void> _load3D(String path) async {
    setState(() { _loading3D = true; _mbError = null; });
    try {
      final f = File(path);
      if (!await f.exists()) throw 'out_3d.json not found at \$path';
      final obj = jsonDecode(await f.readAsString()) as Map<String, dynamic>;
      final T = (obj['T'] as num?)?.toInt() ?? ((obj['coords_3d'] as List).length);
      final coords = (obj['coords_3d'] as List);
      final joints = coords.isNotEmpty ? (coords.first as List).length : 17;
      double minZ = double.infinity, maxZ = -double.infinity;
      for (final fr in coords) {
        for (final j in (fr as List)) {
          final z = (j as List)[2] as num;
          if (z < minZ) minZ = z.toDouble();
          if (z > maxZ) maxZ = z.toDouble();
        }
      }
      setState(() {
        _mbSummary = {
          'frames': T,
          'joints': joints,
          'zMin': minZ,
          'zMax': maxZ,
          'path': path,
        };
      });
    } catch (e) {
      setState(() { _mbError = e.toString(); });
    } finally {
      if (mounted) setState(() { _loading3D = false; });
    }
  }

  Future<void> _retry3D() async {
    final sid = _sessionId;
    if (sid == null) return;
    setState(() { _retrying3D = true; _mbError = null; });
    try {
      final docs = await getApplicationDocumentsDirectory();
      final dir = Directory('${docs.path}/FitPerfect/$sid');
      final file = File('${dir.path}/coco_2d.jsonl');
      if (!file.existsSync()) throw 'coco_2d.jsonl missing in \$sid';
      final lines = await file.readAsLines();
      int w = 640, h = 480;
      for (final ln in lines) {
        if (ln.trim().isEmpty) continue;
        final obj = jsonDecode(ln) as Map<String, dynamic>;
        if (obj.containsKey('img_w') && obj.containsKey('img_h')) {
          w = (obj['img_w'] as num).toInt();
          h = (obj['img_h'] as num).toInt();
          break;
        }
      }
      final ctl = PoseProcessingController();
      final out = await ctl.run3DForSession(sid, Size(w.toDouble(), h.toDouble()));
      if (out != null) {
        _out3dPath = out.path;
        await _load3D(out.path);
      } else {
        throw '3D run returned null';
      }
    } catch (e) {
      if (!mounted) return;
      setState(() { _mbError = 'Retry failed: $e'; });
    } finally {
      if (mounted) setState(() { _retrying3D = false; });
    }
  }

  Widget _build3DSection(BuildContext context) {
    final theme = Theme.of(context);
    final has3d = _out3dPath != null;
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      child: Card(
        elevation: 1,
        child: Padding(
          padding: const EdgeInsets.all(12.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  const Icon(Icons.auto_graph),
                  const SizedBox(width: 8),
                  Text('3D Analysis', style: theme.textTheme.titleMedium),
                  const Spacer(),
                  if (_loading3D || _retrying3D) const SizedBox(
                    width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2)),
                ],
              ),
              const SizedBox(height: 8),
              if (has3d && _mbSummary != null) ...[
                Text('Frames: ${_mbSummary!['frames']}  ·  Joints: ${_mbSummary!['joints']}'),
                Text(
                  'Z range: '
                  '${((_mbSummary!['zMin'] as num?)?.toStringAsFixed(3)) ?? '—'} .. '
                  '${((_mbSummary!['zMax'] as num?)?.toStringAsFixed(3)) ?? '—'}',
                ),
                const SizedBox(height: 8),
                Row(
                  children: [
                    FilledButton.icon(
                      onPressed: () {
                        ScaffoldMessenger.of(context).showSnackBar(
                          const SnackBar(content: Text('3D viewer coming soon')),
                        );
                      },
                      icon: const Icon(Icons.view_in_ar),
                      label: const Text('Open 3D viewer'),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        '${_mbSummary!['path']}',
                        style: theme.textTheme.bodySmall,
                        overflow: TextOverflow.ellipsis,
                      ),
                    )
                  ],
                ),
              ] else ...[
                if (_mbError != null) Text(_mbError!, style: TextStyle(color: theme.colorScheme.error)),
                if (_logTail.isNotEmpty) ...[
                  const Text('Recent 2D log (tail of coco_2d.jsonl):'),
                  Container(
                    constraints: const BoxConstraints(maxHeight: 140),
                    padding: const EdgeInsets.all(8),
                    margin: const EdgeInsets.only(top: 6),
                    color: theme.colorScheme.surfaceVariant.withOpacity(0.4),
                    child: SingleChildScrollView(
                      child: Text(_logTail.join('\n'),
                        style: const TextStyle(fontFamily: 'monospace', fontSize: 11)),
                    ),
                  ),
                  const SizedBox(height: 8),
                ],
                Align(
                  alignment: Alignment.centerLeft,
                  child: OutlinedButton.icon(
                    onPressed: _retrying3D ? null : _retry3D,
                    icon: const Icon(Icons.refresh),
                    label: const Text('Retry 3D'),
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }

  /* ───────── overlay UI helper ───────── */

  Widget _buildOverlay(List<_ErrorEvent> evts, BuildContext ctx) {
    final theme = Theme.of(ctx);
    return Material(
      elevation: 4,
      borderRadius: BorderRadius.circular(8),
      color: theme.colorScheme.surface.withOpacity(0.9),
      child: Padding(
        padding: const EdgeInsets.all(8.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: evts.map((e) {
            final sevColor = switch (e.severity) {
              'severe' => Colors.red,
              'mild'   => Colors.orange,
              _        => Colors.green,
            };
            return Text(
              '${e.label}: ${e.value}  ',
              style: theme.textTheme.bodySmall?.copyWith(color: sevColor),
            );
          }).toList(),
        ),
      ),
    );
  }

  String _fmt(Duration d) =>
      '${d.inMinutes.remainder(60).toString().padLeft(2, '0')}:'
      '${d.inSeconds.remainder(60).toString().padLeft(2, '0')}';
}
