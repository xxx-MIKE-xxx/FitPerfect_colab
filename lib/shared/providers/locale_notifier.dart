import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';

class LocaleNotifier extends ChangeNotifier {
  Locale? _locale;
  Locale? get locale => _locale;

  static const _key = 'locale_code';

  Future<void> loadSavedLocale() async {
    final prefs = await SharedPreferences.getInstance();
    final code = prefs.getString(_key);
    if (code != null) _locale = Locale(code);
    notifyListeners();
  }

  Future<void> setLocale(Locale locale) async {
    _locale = locale;
    notifyListeners();
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_key, locale.languageCode);
  }
}

/// Expose the notifier through Riverpod so every screen can use it.
final localeNotifierProvider = ChangeNotifierProvider<LocaleNotifier>(
  (ref) => LocaleNotifier()..loadSavedLocale(),
);
