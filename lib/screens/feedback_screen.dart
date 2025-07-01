// lib/screens/feedback_screen.dart
// ─────────────────────────────────────────────────────────
import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

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
  }

  @override
  void dispose() {
    _vc.dispose();
    super.dispose();
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
        ],
      ),
      // ‼️  No floatingActionButton – save button removed
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
