// lib/screens/feedback_summary_screen.dart
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:ui' show Size;

import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:path_provider/path_provider.dart';

import '../shared/services/pose_processing_controller.dart';
import '../shared/services/summary_repository.dart';

class FeedbackSummaryScreen extends StatefulWidget {
  const FeedbackSummaryScreen({
    super.key,
    required this.exerciseId,
    this.s3Key,
    this.localReport,
  });

  final String exerciseId;
  final String? s3Key;
  final Map<String, dynamic>? localReport;

  @override
  State<FeedbackSummaryScreen> createState() => _FeedbackSummaryScreenState();
}

class _FeedbackSummaryScreenState extends State<FeedbackSummaryScreen> {
  Map<String, dynamic>? _summary;
  Object? _error;
  String _filter = 'all'; // all | severe | mild
  String? _sessionId;
  String? _out3dPath;
  Map<String, dynamic>? _mbSummary;
  List<String> _logTail = const [];
  bool _loading3D = false;
  bool _retrying3D = false;
  String? _mbError;

  @override
  void initState() {
    super.initState();
    _load();

    final rep = widget.localReport;
    final repSession = rep?['sessionId'];
    final repOut3d = rep?['out3dPath'];
    if (repSession is String) {
      _sessionId = repSession;
    } else if (repSession != null) {
      _sessionId = repSession.toString();
    }
    if (repOut3d is String) {
      _out3dPath = repOut3d;
    } else if (repOut3d != null) {
      _out3dPath = repOut3d.toString();
    }

    if (_out3dPath != null) {
      _loading3D = true;
      _load3D(_out3dPath!);
    } else {
      _loadLogTail();
    }
  }

  Future<void> _load() async {
    try {
      final json = await SummaryRepository.load(
        s3Key: widget.s3Key,
        exerciseId: widget.exerciseId,
        localReport: widget.localReport,
      );
      if (!mounted) return;
      final summarySession = json['sessionId'];
      final summaryOut3d = json['out3dPath'];
      setState(() {
        _summary = json;
        _sessionId ??=
            summarySession is String ? summarySession : summarySession?.toString();
        _out3dPath ??=
            summaryOut3d is String ? summaryOut3d : summaryOut3d?.toString();
      });

      if (_out3dPath != null &&
          (_mbSummary == null || _mbSummary?['path'] != _out3dPath)) {
        _load3D(_out3dPath!);
      } else if (_out3dPath == null && _sessionId != null && _logTail.isEmpty) {
        _loadLogTail();
      }
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e);
    }
  }

  Future<void> _loadLogTail() async {
    final sid = _sessionId;
    if (sid == null) return;

    try {
      final docs = await getApplicationDocumentsDirectory();
      final file = File('${docs.path}/FitPerfect/$sid/coco_2d.jsonl');
      if (!await file.exists()) {
        if (!mounted) return;
        setState(() {
          _logTail = const ['coco_2d.jsonl not found'];
        });
        return;
      }

      final lines = await file.readAsLines();
      final tailCount = 10;
      final tail = lines.length <= tailCount
          ? lines
          : lines.sublist(lines.length - tailCount, lines.length);
      if (!mounted) return;
      setState(() {
        _logTail = tail;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _logTail = ['Error reading coco_2d.jsonl: $e'];
      });
    }
  }

  Future<void> _load3D(String path) async {
    if (mounted) {
      setState(() {
        _loading3D = true;
        _mbError = null;
      });
    } else {
      _loading3D = true;
      _mbError = null;
    }

    try {
      final file = File(path);
      if (!await file.exists()) {
        throw 'out_3d.json missing at $path';
      }

      final str = await file.readAsString();
      final data = jsonDecode(str) as Map<String, dynamic>;
      final coords = data['coords_3d'] as List<dynamic>?;
      if (coords == null || coords.isEmpty) {
        throw 'coords_3d missing or empty';
      }

      final frames = coords.length;
      final firstFrame = coords.first;
      final joints = firstFrame is List ? firstFrame.length : 0;
      double minZ = double.infinity;
      double maxZ = -double.infinity;

      for (final frame in coords) {
        if (frame is! List) continue;
        for (final joint in frame) {
          if (joint is! List || joint.length < 3) continue;
          final z = (joint[2] as num).toDouble();
          if (z < minZ) minZ = z;
          if (z > maxZ) maxZ = z;
        }
      }

      final summary = <String, dynamic>{
        'frames': frames,
        'joints': joints,
        'zMin': minZ.isFinite ? minZ : null,
        'zMax': maxZ.isFinite ? maxZ : null,
        'path': path,
      };

      if (!mounted) return;
      setState(() {
        _out3dPath = path;
        _mbSummary = summary;
        _loading3D = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _mbSummary = null;
        _loading3D = false;
        _mbError = 'Failed to load 3D analysis: $e';
      });
    }
  }

  Future<void> _retry3D() async {
    final sid = _sessionId;
    if (sid == null || _retrying3D) return;

    if (mounted) {
      setState(() {
        _retrying3D = true;
        _mbError = null;
        _loading3D = true;
      });
    } else {
      _retrying3D = true;
      _mbError = null;
      _loading3D = true;
    }

    try {
      final docs = await getApplicationDocumentsDirectory();
      final dir = Directory('${docs.path}/FitPerfect/$sid');
      final file = File('${dir.path}/coco_2d.jsonl');
      if (!await file.exists()) {
        throw 'coco_2d.jsonl missing for session $sid';
      }

      final lines = await file.readAsLines();
      int? imgW;
      int? imgH;
      for (final ln in lines) {
        final trimmed = ln.trim();
        if (trimmed.isEmpty) continue;
        try {
          final obj = jsonDecode(trimmed) as Map<String, dynamic>;
          if (obj.containsKey('img_w') && obj.containsKey('img_h')) {
            imgW = (obj['img_w'] as num).toInt();
            imgH = (obj['img_h'] as num).toInt();
            break;
          }
        } catch (_) {
          continue;
        }
      }

      final ctl = PoseProcessingController();
      final out = await ctl.run3DForSession(
        sid,
        Size((imgW ?? 640).toDouble(), (imgH ?? 480).toDouble()),
      );

      if (out == null) {
        throw '3D processing returned null';
      }

      _out3dPath = out.path;
      await _load3D(out.path);
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _mbError = 'Retry failed: $e';
      });
    } finally {
      if (!mounted) return;
      setState(() {
        _retrying3D = false;
        _loading3D = false;
      });
    }
  }

  Widget _build3DSection(BuildContext context) {
    final theme = Theme.of(context);
    final summary = _mbSummary;
    if (summary == null) {
      return const SizedBox.shrink();
    }

    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
      child: Card(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  const Icon(Icons.auto_graph),
                  const SizedBox(width: 8),
                  Text('3D Analysis', style: theme.textTheme.titleMedium),
                  const Spacer(),
                  if (_loading3D || _retrying3D)
                    const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    ),
                ],
              ),
              const SizedBox(height: 12),
              Text('Frames: ${summary['frames']}'),
              Text('Joints: ${summary['joints']}'),
              Text(
                'Z range: '
                '${(summary['zMin'] as num?)?.toStringAsFixed(3) ?? '—'} .. '
                '${(summary['zMax'] as num?)?.toStringAsFixed(3) ?? '—'}',
              ),
              const SizedBox(height: 12),
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
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
                      '${summary['path']}',
                      style: theme.textTheme.bodySmall,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                ],
              ),
              if (_mbError != null) ...[
                const SizedBox(height: 12),
                Text(
                  _mbError!,
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: theme.colorScheme.error,
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }

  Widget _build3DRetrySection(BuildContext context) {
    final theme = Theme.of(context);
    final canRetry = _sessionId != null && !_retrying3D;

    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
      child: Card(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  const Icon(Icons.threed_rotation),
                  const SizedBox(width: 8),
                  Text('3D Analysis', style: theme.textTheme.titleMedium),
                  const Spacer(),
                  if (_retrying3D || _loading3D)
                    const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    ),
                ],
              ),
              const SizedBox(height: 12),
              Text(
                _sessionId == null
                    ? 'Session information unavailable.'
                    : 'No 3D result yet for session $_sessionId.',
              ),
              const SizedBox(height: 12),
              FilledButton.icon(
                onPressed: canRetry ? _retry3D : null,
                icon: const Icon(Icons.refresh),
                label: Text(_retrying3D ? 'Retrying…' : 'Retry 3D'),
              ),
              if (_mbError != null) ...[
                const SizedBox(height: 12),
                Text(
                  _mbError!,
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: theme.colorScheme.error,
                  ),
                ),
              ],
              const SizedBox(height: 12),
              Text(
                'coco_2d.jsonl tail',
                style: theme.textTheme.bodySmall?.copyWith(fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 8),
              Container(
                width: double.infinity,
                decoration: BoxDecoration(
                  color: theme.colorScheme.surfaceVariant.withOpacity(0.4),
                  borderRadius: BorderRadius.circular(8),
                ),
                padding: const EdgeInsets.all(12),
                child: Text(
                  _logTail.isEmpty
                      ? 'No log data available.'
                      : _logTail.join('\n'),
                  style: theme.textTheme.bodySmall?.copyWith(
                    fontFamily: 'monospace',
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final data = _summary;

    return Scaffold(
      appBar: AppBar(title: const Text('Feedback summary')),
      body: data == null
          ? const Center(child: CircularProgressIndicator())
          : Column(
              children: [
                if (_out3dPath != null && _mbSummary != null) ...[
                  _build3DSection(context),
                ] else ...[
                  _build3DRetrySection(context),
                ],
                Expanded(
                  child: _Content(
                    exerciseId: widget.exerciseId,
                    summary: data,
                    filter: _filter,
                    onFilterChanged: (f) => setState(() => _filter = f),
                  ),
                ),
              ],
            ),
    );
  }
}

/* ───────────────────────────────────────────────────────────────────────── */

class _Content extends StatelessWidget {
  const _Content({
    required this.exerciseId,
    required this.summary,
    required this.filter,
    required this.onFilterChanged,
  });

  final String exerciseId;
  final Map<String, dynamic> summary;
  final String filter;
  final ValueChanged<String> onFilterChanged;

  List<Map<String, dynamic>> _filtered(List<Map<String, dynamic>> all) {
    return switch (filter) {
      'severe' => all.where((m) => m['severity'] == 'severe').toList(),
      'mild'   => all.where((m) => m['severity'] == 'mild').toList(),
      _        => all,
    };
  }

  @override
  Widget build(BuildContext context) {
    final score    = (summary['score'] as num?)?.toInt() ?? 80;
    final mistakes = List<Map<String, dynamic>>.from(summary['mistakes'] ?? []);
    final severeN  = mistakes.where((m) => m['severity'] == 'severe').length;
    final mildN    = mistakes.where((m) => m['severity'] == 'mild').length;
    final allN     = mistakes.length;

    return Column(
      children: [
        const SizedBox(height: 8),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16),
          child: _ScoreCard(score: score),
        ),
        const SizedBox(height: 12),

        // Filter chips
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16),
          child: Wrap(
            spacing: 12,
            runSpacing: 8,
            children: [
              _filterChip(context, 'All ($allN)', 'all', filter, onFilterChanged),
              _filterChip(context, 'Severe ($severeN)', 'severe', filter, onFilterChanged),
              _filterChip(context, 'Mild ($mildN)', 'mild', filter, onFilterChanged),
            ],
          ),
        ),
        const SizedBox(height: 12),

        // Mistakes list / empty state
        Expanded(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            child: _MistakeList(mistakes: _filtered(mistakes)),
          ),
        ),

        // CTAs
        SafeArea(
          top: false,
          minimum: const EdgeInsets.fromLTRB(16, 4, 16, 12),
          child: _Ctas(
            onRecordAgain: () => context.go('/exercises/$exerciseId'),
            onProgress:    () => context.go('/progress'),
            fireGradient: const [Color(0xFFFFC107), Color(0xFFFF7043)], // amber → fire red
            label: 'Record again ($exerciseId)',
          ),
        ),
      ],
    );
  }

  Widget _filterChip(
    BuildContext ctx,
    String label,
    String value,
    String current,
    ValueChanged<String> onChanged,
  ) {
    final selected = value == current;
    return ChoiceChip(
      label: Text(label),
      selected: selected,
      onSelected: (_) => onChanged(value),
      labelStyle: TextStyle(
        fontWeight: FontWeight.w600,
        color: selected ? Theme.of(ctx).colorScheme.onPrimary : null,
      ),
      selectedColor: Theme.of(ctx).colorScheme.primary,
      backgroundColor: Theme.of(ctx).colorScheme.surface,
      shape: StadiumBorder(
        side: BorderSide(
          color: selected
              ? Theme.of(ctx).colorScheme.primary
              : Theme.of(ctx).dividerColor.withOpacity(.35),
        ),
      ),
    );
  }
}

/* ───────────────────────── SCORE CARD (FIERY) ───────────────────────── */

class _ScoreCard extends StatelessWidget {
  const _ScoreCard({required this.score});
  final int score;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final pct = (score / 100.0).clamp(0.0, 1.0);

    // Always warm, energetic dial colors (amber → fire red)
    const ringGradient = [
      Color(0xFFFFC107), // amber
      Color(0xFFFF8F00), // deep amber
      Color(0xFFFF7043), // orange red
      Color(0xFFFF3D00), // fire red
    ];

    return Material(
      elevation: 8,
      color: theme.colorScheme.surface,
      borderRadius: BorderRadius.circular(24),
      shadowColor: Colors.black12,
      child: Container(
        padding: const EdgeInsets.all(18),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(24),
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              theme.colorScheme.surface,
              theme.colorScheme.surface.withOpacity(.96),
            ],
          ),
        ),
        child: Row(
          children: [
            // Dial with solid white puck for ultra-clear number
            SizedBox(
              width: 96,
              height: 96,
              child: Stack(
                alignment: Alignment.center,
                children: [
                  CustomPaint(
                    size: const Size.square(96),
                    painter: _DialPainter(
                      progress: pct,
                      trackColor: theme.colorScheme.outlineVariant.withOpacity(.30),
                      gradient: SweepGradient(
                        startAngle: -math.pi / 2,
                        endAngle:  3 * math.pi / 2,
                        colors: ringGradient,
                      ),
                    ),
                  ),
                  Container(
                    width: 64,
                    height: 64,
                    decoration: const BoxDecoration(
                      color: Colors.white,
                      shape: BoxShape.circle,
                      boxShadow: [
                        BoxShadow(
                          blurRadius: 10,
                          offset: Offset(0, 2),
                          color: Color(0x1A000000),
                        ),
                      ],
                    ),
                    alignment: Alignment.center,
                    child: Text(
                      '$score',
                      style: theme.textTheme.titleLarge?.copyWith(
                        fontWeight: FontWeight.w900,
                        letterSpacing: -0.5,
                        color: Colors.black87,
                      ),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(width: 16),

            // Text block + festive dots BELOW text (won’t cover score)
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Overall score', style: theme.textTheme.labelMedium),
                  const SizedBox(height: 4),
                  Text(
                    _headline(score),
                    style: theme.textTheme.headlineSmall?.copyWith(
                      fontWeight: FontWeight.w900,
                      letterSpacing: -0.3,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    _subline(score),
                    style: theme.textTheme.bodyMedium?.copyWith(
                      color: theme.textTheme.bodySmall?.color?.withOpacity(.8),
                    ),
                  ),
                  const SizedBox(height: 10),
                  Row(
                    children: List.generate(10, (i) {
                      final colors = [
                        const Color(0xFFFFC107),
                        const Color(0xFFFF7043),
                        const Color(0xFF42A5F5),
                        const Color(0xFF66BB6A),
                        const Color(0xFFFFEB3B),
                      ];
                      return Padding(
                        padding: const EdgeInsets.only(right: 6),
                        child: Container(
                          width: 6, height: 6,
                          decoration: BoxDecoration(
                            color: colors[i % colors.length].withOpacity(.95),
                            shape: BoxShape.circle,
                          ),
                        ),
                      );
                    }),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  String _headline(int s) {
    if (s >= 90) return 'Excellent form';
    if (s >= 80) return 'Strong form';
    if (s >= 70) return 'Good baseline';
    if (s >= 60) return 'Needs attention';
    return 'Work in progress';
  }

  String _subline(int s) {
    if (s >= 90) return 'Minor tweaks only';
    if (s >= 80) return 'Just a few adjustments';
    if (s >= 70) return 'Fix the key issues below';
    if (s >= 60) return 'Focus on fundamentals';
    return 'Let’s rebuild form step by step';
  }
}

class _DialPainter extends CustomPainter {
  _DialPainter({
    required this.progress,
    required this.trackColor,
    required this.gradient,
  });

  final double progress; // 0..1
  final Color trackColor;
  final Gradient gradient;

  @override
  void paint(Canvas canvas, Size size) {
    final center = size.center(Offset.zero);
    final radius = size.width / 2;

    final track = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 10
      ..color = trackColor
      ..strokeCap = StrokeCap.round;

    final prog = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 10
      ..strokeCap = StrokeCap.round
      ..shader = gradient.createShader(
        Rect.fromCircle(center: center, radius: radius),
      );

    final rect = Rect.fromCircle(center: center, radius: radius - 5);
    canvas.drawArc(rect, -math.pi / 2, 2 * math.pi, false, track);
    canvas.drawArc(rect, -math.pi / 2, (2 * math.pi) * progress, false, prog);
  }

  @override
  bool shouldRepaint(covariant _DialPainter old) =>
      old.progress != progress || old.trackColor != trackColor;
}

/* ───────────────────────── MISTAKES ───────────────────────── */

class _MistakeList extends StatelessWidget {
  const _MistakeList({required this.mistakes});
  final List<Map<String, dynamic>> mistakes;

  Color _sevColor(BuildContext ctx, String s) => switch (s) {
        'severe' => const Color(0xFFE53935), // red
        'mild'   => const Color(0xFFFF8F00), // amber
        _        => Theme.of(ctx).colorScheme.primary,
      };

  @override
  Widget build(BuildContext context) {
    if (mistakes.isEmpty) {
      return _EmptyState();
    }

    return ListView.separated(
      itemCount: mistakes.length,
      separatorBuilder: (_, __) => const SizedBox(height: 10),
      padding: const EdgeInsets.only(bottom: 12, top: 4),
      itemBuilder: (ctx, i) {
        final m = mistakes[i];
        final sev = (m['severity'] as String?) ?? 'mild';
        final color = _sevColor(ctx, sev);

        return Container(
          decoration: BoxDecoration(
            color: Theme.of(ctx).colorScheme.surface,
            borderRadius: BorderRadius.circular(16),
            boxShadow: const [
              BoxShadow(
                color: Color(0x0F000000),
                blurRadius: 16,
                offset: Offset(0, 6),
              )
            ],
          ),
          child: ListTile(
            leading: Container(
              width: 42, height: 42,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                gradient: LinearGradient(
                  colors: [
                    color.withOpacity(.15),
                    color.withOpacity(.30),
                  ],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
              ),
              child: Icon(
                sev == 'severe' ? Icons.error : Icons.warning_amber_rounded,
                color: color,
              ),
            ),
            title: Text(
              (m['title'] as String?) ?? 'Form issue',
              style: Theme.of(ctx).textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.w800,
                  ),
            ),
            subtitle: (m['detail'] as String?) == null
                ? null
                : Text(m['detail'] as String),
            trailing: Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
              decoration: BoxDecoration(
                color: color.withOpacity(0.12),
                borderRadius: BorderRadius.circular(99),
                border: Border.all(color: color.withOpacity(0.3)),
              ),
              child: Text(
                sev[0].toUpperCase() + sev.substring(1),
                style: TextStyle(color: color, fontWeight: FontWeight.w700),
              ),
            ),
          ),
        );
      },
    );
  }
}

class _EmptyState extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.emoji_events, size: 56, color: theme.colorScheme.primary),
          const SizedBox(height: 14),
          Text(
            'No mistakes detected. Great job!',
            textAlign: TextAlign.center,
            style: theme.textTheme.titleLarge?.copyWith(
              fontWeight: FontWeight.w900,
              letterSpacing: -.2,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'Keep up the form! You can record again to aim even higher.',
            textAlign: TextAlign.center,
            style: theme.textTheme.bodyLarge?.copyWith(
              color: theme.textTheme.bodySmall?.color?.withOpacity(.8),
            ),
          ),
        ],
      ),
    );
  }
}

/* ───────────────────────── CTAs ───────────────────────── */

class _Ctas extends StatelessWidget {
  const _Ctas({
    required this.onRecordAgain,
    required this.onProgress,
    required this.fireGradient,
    required this.label,
  });

  final VoidCallback onRecordAgain;
  final VoidCallback onProgress;
  final List<Color> fireGradient;
  final String label;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Material(
      elevation: 8,
      color: theme.colorScheme.surface,
      borderRadius: BorderRadius.circular(24),
      child: Padding(
        padding: const EdgeInsets.fromLTRB(12, 12, 12, 12),
        child: Column(
          children: [
            // Fiery gradient primary CTA
            DecoratedBox(
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: fireGradient,
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                borderRadius: BorderRadius.circular(28),
              ),
              child: SizedBox(
                width: double.infinity,
                height: 56,
                child: TextButton(
                  style: TextButton.styleFrom(
                    foregroundColor: Colors.white,
                    shape: const StadiumBorder(),
                  ),
                  onPressed: onRecordAgain,
                  child: Text(
                    label,
                    style: const TextStyle(
                      fontWeight: FontWeight.w900,
                      fontSize: 18,
                      letterSpacing: .1,
                    ),
                  ),
                ),
              ),
            ),
            const SizedBox(height: 12),
            // Secondary CTA
            SizedBox(
              width: double.infinity,
              height: 56,
              child: OutlinedButton(
                style: OutlinedButton.styleFrom(
                  shape: const StadiumBorder(),
                  side: BorderSide(color: theme.colorScheme.primary),
                ),
                onPressed: onProgress,
                child: Text(
                  'Continue to Progress',
                  style: TextStyle(
                    fontWeight: FontWeight.w800,
                    color: theme.colorScheme.primary,
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
