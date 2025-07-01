// lib/screens/profile_screen.dart
import 'package:flutter/material.dart';
import 'package:flutter_gen/gen_l10n/app_localizations.dart';
import 'package:go_router/go_router.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../shared/providers/auth_provider.dart';          // ← for sign‑out

class ProfileScreen extends ConsumerWidget {
  const ProfileScreen({super.key});

  /* ───────────── UI ───────────── */
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final t = AppLocalizations.of(context)!;

    Future<void> _handleSignOut() async {
      await ref.read(authProvider.notifier).signOut();     // ← single source
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Signed‑out – restart the flow 👋')),
        );
      }
    }

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        _section(
          context,
          t.general,
          [
            (label: t.editProfile, route: 'edit-profile'),
            (label: t.notifications, route: 'notifications'),
            (label: t.languageUnits, route: 'language-units'),
          ],
        ),
        _section(
          context,
          t.recordingAnalysis,
          [
            (label: t.videoQuality, route: 'video-quality'),
            (label: t.aiSensitivity, route: 'ai-sensitivity'),
            (label: t.saveAutoDelete, route: 'save-auto-delete'),
          ],
        ),
        _section(
          context,
          t.dataProgress,
          [
            (label: t.trainingHistory, route: 'training-history'),
            (label: t.exportData, route: 'export-data'),
            (label: t.syncFitness, route: 'sync-fitness'),
          ],
        ),
        _section(
          context,
          t.privacySecurity,
          [
            (label: t.privacyControls, route: 'privacy-controls'),
            (label: t.accountSecurity, route: 'account-security'),
          ],
        ),
        const SizedBox(height: 24),
        FilledButton(
          style: FilledButton.styleFrom(backgroundColor: Colors.red),
          onPressed: _handleSignOut,
          child: Text(t.logout),
        ),
      ],
    );
  }

  /// Helper that renders one titled section of tiles.
  Widget _section(
    BuildContext context,
    String title,
    List<({String label, String route})> items,
  ) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const SizedBox(height: 24),
        Text(title, style: const TextStyle(fontWeight: FontWeight.bold)),
        ...items.map(
          (item) => ListTile(
            title: Text(item.label),
            trailing: const Icon(Icons.chevron_right),
            onTap: () => context.pushNamed(item.route),
          ),
        ),
      ],
    );
  }
}
