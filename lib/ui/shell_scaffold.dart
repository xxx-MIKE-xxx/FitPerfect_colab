// ───────────────────────── lib/ui/shell_scaffold.dart ─────────────────────────
//
// A simple “container” widget that shows the current page (child) and a bottom
// navigation bar to switch between Progress | Exercises | Profile.
//

import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:flutter_gen/gen_l10n/app_localizations.dart';   // ← for translations

class ShellScaffold extends StatelessWidget {
  const ShellScaffold({super.key, required this.child});
  final Widget child;

  // keep routes & icons static ­– only labels will be localised at runtime
  static final _destinations = <_NavDestination>[
    _NavDestination(label: 'Progress', icon: Icons.show_chart,     route: '/progress'),
    _NavDestination(label: 'Exercises', icon: Icons.fitness_center, route: '/exercises'),
    _NavDestination(label: 'Profile',  icon: Icons.person,          route: '/profile'),
  ];

  @override
  Widget build(BuildContext context) {
    final t = AppLocalizations.of(context)!;                       // ← localisation helper

    // map i → translated label
    final _labels = [t.progressTab, t.exercisesTab, t.profileTab];

    final String location = GoRouterState.of(context).uri.toString();
    final int index = _destinations.indexWhere((d) => location.startsWith(d.route));
    final int currentIndex = (index == -1) ? 1 : index; // default to “Exercises”

    return Scaffold(
      body: child,
      bottomNavigationBar: NavigationBar(
        selectedIndex: currentIndex,
        onDestinationSelected: (i) => context.go(_destinations[i].route),
        destinations: List.generate(
          _destinations.length,
          (i) => NavigationDestination(
            icon: Icon(_destinations[i].icon),
            label: _labels[i],                                   // ← translated
          ),
        ),
      ),
    );
  }
}

class _NavDestination {
  const _NavDestination({required this.label, required this.icon, required this.route});
  final String label;
  final IconData icon;
  final String route;
}
