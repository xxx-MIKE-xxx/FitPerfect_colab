// lib/screens/settings/language_units_screen.dart
import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../shared/providers/locale_notifier.dart';
import 'package:flutter_gen/gen_l10n/app_localizations.dart';

class LanguageUnitsScreen extends ConsumerStatefulWidget {
  const LanguageUnitsScreen({super.key});

  @override
  ConsumerState<LanguageUnitsScreen> createState() => _LanguageUnitsScreenState();
}

class _LanguageUnitsScreenState extends ConsumerState<LanguageUnitsScreen> {
  late String language;                 // ← zmieniono: late, bez domyślnej wartości
  String units = 'metric';

  static const languages = [
    ('en', 'English'),
    ('pl', 'Polski'),
  ];

  @override
  void initState() {
    super.initState();
    // pobierz zapisany język z providera; jeśli brak, domyśl się 'en'
    language = ref.read(localeNotifierProvider).locale?.languageCode ?? 'en';
  }

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context)!;

    return Scaffold(
      appBar: AppBar(title: Text(l10n.languageUnits)),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Text(l10n.language, style: const TextStyle(fontWeight: FontWeight.bold)),
          ...languages.map(
            (tuple) => RadioListTile(
              value: tuple.$1,
              groupValue: language,                     // ← odzwierciedla stan
              onChanged: (v) async {
                setState(() => language = v!);
                await ref
                    .read(localeNotifierProvider.notifier)
                    .setLocale(Locale(v!));             // v! – już nie nullable
              },
              title: Text(tuple.$2),
            ),
          ),
          const SizedBox(height: 24),
          Text(l10n.units, style: const TextStyle(fontWeight: FontWeight.bold)),
          RadioListTile(
            value: 'metric',
            groupValue: units,
            onChanged: (v) => setState(() => units = v!),
            title: const Text('Metric (kg, km)'),
          ),
          RadioListTile(
            value: 'imperial',
            groupValue: units,
            onChanged: (v) => setState(() => units = v!),
            title: const Text('Imperial (lb, mi)'),
          ),
        ],
      ),
    );
  }
}
