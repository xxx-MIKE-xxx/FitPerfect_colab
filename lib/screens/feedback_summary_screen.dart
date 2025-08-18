// lib/screens/feedback_summary_screen.dart
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
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

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      final json = await SummaryRepository.load(
        s3Key: widget.s3Key,
        exerciseId: widget.exerciseId,
        localReport: widget.localReport,
      );
      setState(() => _summary = json);
    } catch (e) {
      setState(() => _error = e);
    }
  }

  @override
  Widget build(BuildContext context) {
    final data = _summary;

    return Scaffold(
      appBar: AppBar(title: const Text('Feedback summary')),
      body: data == null
          ? const Center(child: CircularProgressIndicator())
          : _Content(
              exerciseId: widget.exerciseId,
              summary: data,
              filter: _filter,
              onFilterChanged: (f) => setState(() => _filter = f),
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
