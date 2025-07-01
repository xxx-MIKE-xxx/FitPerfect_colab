import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

/// Master list of exercises – (slug, icon, display label)
const _allExercises = [
  ('squat', Icons.fitness_center, 'Squat'),
  ('deadlift', Icons.accessibility_new, 'Deadlift'),
  ('bench', Icons.sports_mma, 'Bench Press'),
  ('pullup', Icons.arrow_circle_up, 'Pull‑up'),
  ('lunges', Icons.directions_run, 'Lunges'),
  ('shoulder', Icons.pan_tool_alt, 'Shoulder Press'),
  // New additions --------------------------------------------------------------
  ('plank', Icons.self_improvement, 'Plank'),
  ('burpee', Icons.fitness_center, 'Burpee'),
  ('bicep_curl', Icons.fitness_center, 'Bicep Curl'),
  ('tricep_dip', Icons.fitness_center, 'Tricep Dip'),
  ('mountain_climber', Icons.landscape_rounded, 'Mountain Climber'),
  ('row', Icons.rowing, 'Row'),
  ('leg_press', Icons.directions_walk, 'Leg Press'),
  ('crunch', Icons.airline_seat_recline_extra, 'Crunch'),
  ('hip_thrust', Icons.airline_seat_flat, 'Hip Thrust'),
  ('calf_raise', Icons.directions_run, 'Calf Raise'),
];

class ExerciseGridScreen extends StatefulWidget {
  const ExerciseGridScreen({super.key});

  @override
  State<ExerciseGridScreen> createState() => _ExerciseGridScreenState();
}

class _ExerciseGridScreenState extends State<ExerciseGridScreen> {
  String _query = '';

  @override
  Widget build(BuildContext context) {
    // Filter exercises by search string (case‑insensitive)
    final exercises = _allExercises
        .where((e) => e.$3.toLowerCase().contains(_query.toLowerCase()))
        .toList();

    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).scaffoldBackgroundColor,
        elevation: 0,
        title: SizedBox(
          height: 40,
          child: TextField(
            onChanged: (v) => setState(() => _query = v),
            decoration: InputDecoration(
              hintText: 'Search exercises…', // l10n key: searchExercises
              filled: true,
              fillColor: const Color(0xFFF2F2F2), // Light grey for subtle contrast
              contentPadding: const EdgeInsets.symmetric(horizontal: 16),
              prefixIcon: _query.isEmpty ? const Icon(Icons.search) : null,
              suffixIcon: _query.isNotEmpty
                  ? IconButton(
                      onPressed: () => setState(() => _query = ''),
                      icon: const Icon(Icons.clear),
                    )
                  : null,
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(24),
                borderSide: BorderSide.none,
              ),
              enabledBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(24),
                borderSide: BorderSide.none,
              ),
              focusedBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(24),
                borderSide: BorderSide.none,
              ),
            ),
          ),
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: GridView.builder(
          itemCount: exercises.length,
          gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
            crossAxisCount: 2,
            mainAxisSpacing: 16,
            crossAxisSpacing: 16,
          ),
          itemBuilder: (_, i) {
            final (slug, icon, label) = exercises[i];
            return InkWell(
              onTap: () => context.go('/exercises/$slug'),
              borderRadius: BorderRadius.circular(20), // Slightly rounder
              child: Container(
                decoration: BoxDecoration(
                  gradient: const LinearGradient(
                    // Updated to a cool blue gradient for a fresh look
                    colors: [Color(0xFF1565C0), Color(0xFF5E92F3)],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(icon, color: Colors.white, size: 48),
                    const SizedBox(height: 8),
                    Text(
                      label,
                      textAlign: TextAlign.center,
                      style: const TextStyle(
                          color: Colors.white,
                          fontSize: 18,
                          fontWeight: FontWeight.w600),
                    ),
                  ],
                ),
              ),
            );
          },
        ),
      ),
    );
  }
}
