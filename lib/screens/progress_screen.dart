import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:share_plus/share_plus.dart';

/// ProgressScreen – shows the user's overall fitness progress, technique tips,
/// workout history timeline, and lets them share their progress.
///
/// All data is currently mocked – wire up to your backend / state‑management
/// solution of choice.
class ProgressScreen extends StatelessWidget {
  ProgressScreen({super.key});

  // Mocked values ----------------------------------------------------------------
  final int _overallScore = 72; // 0–100
  final List<String> _suggestions = const [
    'Keep your back straight during deadlifts',
    'Engage your core throughout planks',
    'Land softly when jumping to reduce knee stress',
  ];

  final List<_WorkoutEntry> _workoutHistory = [
    _WorkoutEntry(
      date: DateTime(2025, 6, 11),
      title: 'Upper‑Body Strength',
      details: 'Bench press PR +2 kg',
    ),
    _WorkoutEntry(
      date: DateTime(2025, 6, 9),
      title: 'HIIT Cardio',
      details: 'Average HR 158 bpm',
    ),
    _WorkoutEntry(
      date: DateTime(2025, 6, 7),
      title: 'Leg Day',
      details: 'Squat depth improved',
    ),
  ];

  // Helper: Share sheet -----------------------------------------------------------
  void _shareScore(BuildContext context) {
    final box = context.findRenderObject() as RenderBox?;
    Share.share(
      'My fitness score is $_overallScore! 💪',
      // Needed for iPad/iOS to know where to anchor the pop‑over
      sharePositionOrigin: box!.localToGlobal(Offset.zero) & box.size,
    );
  }

  // -----------------------------------------------------------------------------

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Progress'),
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Overall Score ------------------------------------------------------
            Text(
              'Overall Score',
              style: Theme.of(context).textTheme.titleLarge,
            ),
            const SizedBox(height: 8),
            _ScoreBar(score: _overallScore.toDouble()),
            const SizedBox(height: 4),
            Text('$_overallScore / 100',
                style: Theme.of(context).textTheme.bodySmall),
            const SizedBox(height: 24),

            // Technique Tips -----------------------------------------------------
            Text(
              'Technique Tips',
              style: Theme.of(context).textTheme.titleLarge,
            ),
            const SizedBox(height: 8),
            Card(
              elevation: 2,
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(16)),
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  children: _suggestions
                      .map((tip) => ListTile(
                            leading: const Icon(Icons.fitness_center_rounded),
                            title: Text(tip),
                          ))
                      .toList(),
                ),
              ),
            ),
            const SizedBox(height: 24),

            // Workout History ----------------------------------------------------
            Text(
              'Workout History',
              style: Theme.of(context).textTheme.titleLarge,
            ),
            const SizedBox(height: 8),
            ListView.separated(
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              itemCount: _workoutHistory.length,
              separatorBuilder: (_, __) => const SizedBox(height: 12),
              itemBuilder: (context, index) {
                final entry = _workoutHistory[index];
                return _TimelineTile(entry: entry, isFirst: index == 0);
              },
            ),
            const SizedBox(height: 80), // space for the FAB
          ],
        ),
      ),
      // Using Builder gives us a fresh BuildContext anchored *inside* the Scaffold,
      // which is required for sharePositionOrigin on iOS iPad pop‑over.
      floatingActionButton: Builder(
        builder: (context) => FloatingActionButton.extended(
          onPressed: () => _shareScore(context),
          icon: const Icon(Icons.share_rounded),
          label: const Text('Share Progress'),
        ),
      ),
    );
  }
}

// ==============================================================================
// Widgets & Models
// ==============================================================================

class _ScoreBar extends StatelessWidget {
  const _ScoreBar({required this.score});
  final double score; // 0–100

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(builder: (context, constraints) {
      final width = constraints.maxWidth * (score.clamp(0, 100) / 100);

      return ClipRRect(
        borderRadius: BorderRadius.circular(12),
        child: Stack(
          children: [
            Container(
              height: 20,
              decoration: BoxDecoration(
                gradient: const LinearGradient(
                  colors: [
                    Colors.red,
                    Colors.orange,
                    Colors.yellow,
                    Colors.green,
                  ],
                ),
              ),
              width: width,
            ),
            Container(
              height: 20,
              width: constraints.maxWidth,
              decoration: BoxDecoration(
                color: Colors.grey.shade300.withOpacity(0.3),
              ),
            ),
          ],
        ),
      );
    });
  }
}

class _WorkoutEntry {
  const _WorkoutEntry({
    required this.date,
    required this.title,
    required this.details,
  });
  final DateTime date;
  final String title;
  final String details;
}

class _TimelineTile extends StatelessWidget {
  const _TimelineTile({
    required this.entry,
    required this.isFirst,
  });
  final _WorkoutEntry entry;
  final bool isFirst;

  @override
  Widget build(BuildContext context) {
    final dateStr = DateFormat.MMMd().format(entry.date);
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Timeline marker ------------------------------------------------------
        Column(
          children: [
            // Dot
            Container(
              width: 12,
              height: 12,
              decoration: BoxDecoration(
                color: Theme.of(context).colorScheme.primary,
                shape: BoxShape.circle,
              ),
            ),
            if (!isFirst)
              Container(
                width: 2,
                height: 60,
                color: Theme.of(context).colorScheme.primary,
              ),
          ],
        ),
        const SizedBox(width: 12),

        // Card with workout info ----------------------------------------------
        Expanded(
          child: Card(
            elevation: 1,
            shape:
                RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(dateStr,
                      style: Theme.of(context).textTheme.labelSmall),
                  const SizedBox(height: 4),
                  Text(entry.title,
                      style: Theme.of(context).textTheme.titleMedium),
                  const SizedBox(height: 4),
                  Text(entry.details),
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }
}
