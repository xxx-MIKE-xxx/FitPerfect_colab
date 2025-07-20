import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'ui/shell_scaffold.dart';
import 'features/auth/login_page.dart';
import 'screens/exercise_grid_screen.dart';
import 'screens/exercise_preview_screen.dart';
import 'screens/progress_screen.dart';
import 'screens/profile_screen.dart';
import 'screens/settings/edit_profile_screen.dart';
import 'screens/settings/notifications_screen.dart';
import 'screens/settings/language_units_screen.dart';
import 'screens/settings/video_quality_screen.dart';
import 'screens/settings/save_auto_delete_screen.dart';
import 'screens/settings/training_history_screen.dart';
import 'screens/settings/export_data_screen.dart';
import 'screens/settings/privacy_controls_screen.dart';
import 'screens/settings/account_security_screen.dart';
import 'screens/feedback_screen.dart'; 

final router = GoRouter(
  routes: [
    ShellRoute(
      builder: (context, state, child) => ShellScaffold(child: child),
      routes: [
        GoRoute(
          path: '/progress',
          pageBuilder: (context, state) =>
              NoTransitionPage(child: ProgressScreen()), // ← removed both consts
        ),
        GoRoute(
          path: '/exercises',
          pageBuilder: (c, s) => const NoTransitionPage(child: ExerciseGridScreen()),
          routes: [
            GoRoute(
              path: ':id', // e.g. /exercises/squat
              pageBuilder: (c, s) => MaterialPage(
                fullscreenDialog: true,
                child: ExercisePreviewScreen(exerciseId: s.pathParameters['id']!),
              ),
            ),
          ],
        ),
        GoRoute(
          path: '/profile',
          pageBuilder: (c, s) => const NoTransitionPage(child: ProfileScreen()),
        ),
        // default redirect
        GoRoute(
          path: '/',
          redirect: (_, __) => '/exercises',
        ),
        

        GoRoute(
          path: '/settings/edit-profile',
          name: 'edit-profile',
          builder: (_, __) => const EditProfileScreen(),
        ),
        GoRoute(
          path: '/settings/notifications',
          name: 'notifications',
          builder: (_, __) => const NotificationsScreen(),
        ),
        GoRoute(
          path: '/settings/language-units',
          name: 'language-units',
          builder: (_, __) => const LanguageUnitsScreen(),
        ),
        GoRoute(
          path: '/settings/video-quality',
          name: 'video-quality',
          builder: (_, __) => const VideoQualityScreen(),
        ),
        
        GoRoute(
          path: '/settings/save-auto-delete',
          name: 'save-auto-delete',
          builder: (_, __) => const SaveAutoDeleteScreen(),
        ),
        GoRoute(
          path: '/settings/training-history',
          name: 'training-history',
          builder: (_, __) => const TrainingHistoryScreen(),
        ),
        GoRoute(
          path: '/settings/export-data',
          name: 'export-data',
          builder: (_, __) => const ExportDataScreen(),
        ),
        GoRoute(
          path: '/settings/privacy-controls',
          name: 'privacy-controls',
          builder: (_, __) => const PrivacyControlsScreen(),
        ),
        GoRoute(
          path: '/settings/account-security',
          name: 'account-security',
          builder: (_, __) => const AccountSecurityScreen(),
        ),
        GoRoute(
          path: '/feedback',
          builder: (ctx, state) {
            final extra   = state.extra! as Map<String, dynamic>;
            return FeedbackScreen(
              videoKey : extra['videoKey'] as String,
              report   : extra['report']   as Map<String, dynamic>,
            );
          },
        ),



      ],
    ),
  ],
);

GoRouter createRouter() => router;