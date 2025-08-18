// lib/screens/exercise_grid_screen.dart
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:go_router/go_router.dart';

/// Master list of exercises – (slug, icon, display label)
const _allExercises = [
  ('squat', Icons.fitness_center, 'Squat'),
  ('deadlift', Icons.accessibility_new, 'Deadlift'),
  ('bench', Icons.sports_mma, 'Bench Press'),
  ('pullup', Icons.arrow_circle_up, 'Pull-up'),
  ('lunges', Icons.directions_run, 'Lunges'),
  ('shoulder', Icons.pan_tool_alt, 'Shoulder Press'),
  // New additions --------------------------------------------------------------
  ('plank', Icons.self_improvement, 'Plank'),
  ('burpee', Icons.fitness_center, 'Burpee'),
  ('bicep_curl', Icons.fitness_center, 'Bicep Curl'),
  ('tricep_dip', Icons.fitness_center, 'Tricep Dip'),
  ('mountain_climber', Icons.landscape_rounded, 'Mountain Climber'),
  ('row', Icons.rowing, 'Row'),
  ('leg_press', Icons.directions_walk, 'Leg Press'),
  ('crunch', Icons.airline_seat_recline_extra, 'Crunch'),
  ('hip_thrust', Icons.airline_seat_flat, 'Hip Thrust'),
  ('calf_raise', Icons.directions_run, 'Calf Raise'),
];

/// A small palette of energetic gradients we’ll cycle through for the tiles.
const _tileGradients = <List<Color>>[
  [Color(0xFFFFC107), Color(0xFFFF7043)], // amber → fire red
  [Color(0xFF00BFA5), Color(0xFF1DE9B6)], // teal → mint
  [Color(0xFF42A5F5), Color(0xFF7E57C2)], // blue → purple
  [Color(0xFFFF5F6D), Color(0xFFFFC371)], // coral → sunrise
  [Color(0xFF00E5FF), Color(0xFF2979FF)], // aqua → royal blue
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
    // Filter exercises by search string (case-insensitive)
    final exercises = _allExercises
        .where((e) => e.$3.toLowerCase().contains(_query.toLowerCase()))
        .toList();

    final theme = Theme.of(context);

    return Scaffold(
      backgroundColor: theme.colorScheme.surface,
      appBar: AppBar(
        elevation: 0,
        backgroundColor: theme.colorScheme.surface,
        titleSpacing: 16,
        title: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Exercises',
              style: theme.textTheme.headlineSmall?.copyWith(
                fontWeight: FontWeight.w800,
                letterSpacing: -0.3,
              ),
            ),
            const SizedBox(height: 12),
            // Premium-looking search bar
            Material(
              color: Colors.white,
              elevation: 6,
              shadowColor: Colors.black.withOpacity(.08),
              borderRadius: BorderRadius.circular(28),
              child: SizedBox(
                height: 44,
                child: TextField(
                  onChanged: (v) => setState(() => _query = v),
                  textInputAction: TextInputAction.search,
                  decoration: InputDecoration(
                    hintText: 'Search exercises…',
                    hintStyle: TextStyle(
                      color: theme.textTheme.bodyMedium?.color?.withOpacity(.55),
                    ),
                    prefixIcon: const Icon(Icons.search),
                    suffixIcon: _query.isNotEmpty
                        ? IconButton(
                            tooltip: 'Clear',
                            onPressed: () => setState(() => _query = ''),
                            icon: const Icon(Icons.clear),
                          )
                        : null,
                    border: InputBorder.none,
                    contentPadding:
                        const EdgeInsets.symmetric(horizontal: 16, vertical: 0),
                  ),
                ),
              ),
            ),
          ],
        ),
        toolbarHeight: 112,
      ),
      body: Padding(
        padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
        child: GridView.builder(
          itemCount: exercises.length,
          gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
            crossAxisCount: 2,
            mainAxisSpacing: 16,
            crossAxisSpacing: 16,
            childAspectRatio: 1.05, // slightly taller for visual balance
          ),
          itemBuilder: (_, i) {
            final (slug, icon, label) = exercises[i];
            final colors = _tileGradients[i % _tileGradients.length];
            return _ExerciseTile(
              slug: slug,
              icon: icon,
              label: label,
              colors: colors,
            );
          },
        ),
      ),
    );
  }
}

/* ───────────────────────────── Tile ───────────────────────────── */

class _ExerciseTile extends StatefulWidget {
  const _ExerciseTile({
    required this.slug,
    required this.icon,
    required this.label,
    required this.colors,
  });

  final String slug;
  final IconData icon;
  final String label;
  final List<Color> colors;

  @override
  State<_ExerciseTile> createState() => _ExerciseTileState();
}

class _ExerciseTileState extends State<_ExerciseTile> {
  bool _pressed = false;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return AnimatedScale(
      duration: const Duration(milliseconds: 120),
      scale: _pressed ? 0.98 : 1.0,
      child: InkWell(
        onTapDown: (_) => setState(() => _pressed = true),
        onTapCancel: () => setState(() => _pressed = false),
        onTap: () {
          HapticFeedback.selectionClick();
          setState(() => _pressed = false);
          context.go('/exercises/${widget.slug}');
        },
        borderRadius: BorderRadius.circular(22),
        child: Container(
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(22),
            gradient: LinearGradient(
              colors: widget.colors,
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
            boxShadow: [
              BoxShadow(
                color: widget.colors.last.withOpacity(.35),
                blurRadius: 24,
                offset: const Offset(0, 10),
              ),
            ],
          ),
          child: Stack(
            children: [
              // Decorative glossy sweep
              Positioned(
                top: -24,
                left: -18,
                child: Container(
                  width: 120,
                  height: 120,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    gradient: LinearGradient(
                      colors: [
                        Colors.white.withOpacity(.28),
                        Colors.white.withOpacity(.06),
                      ],
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                    ),
                  ),
                ),
              ),
              // Content
              Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    // Icon puck
                    Container(
                      width: 72,
                      height: 72,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: Colors.white.withOpacity(.20),
                        border: Border.all(
                          color: Colors.white.withOpacity(.30),
                          width: 1.2,
                        ),
                        boxShadow: const [
                          BoxShadow(
                            color: Color(0x14000000),
                            blurRadius: 10,
                            offset: Offset(0, 4),
                          ),
                        ],
                      ),
                      child: Icon(widget.icon, color: Colors.white, size: 36),
                    ),
                    const SizedBox(height: 12),
                    Text(
                      widget.label,
                      textAlign: TextAlign.center,
                      style: theme.textTheme.titleMedium?.copyWith(
                        color: Colors.white,
                        fontWeight: FontWeight.w800,
                        letterSpacing: -.2,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
