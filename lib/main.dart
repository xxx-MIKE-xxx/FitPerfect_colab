// lib/main.dart
import 'package:flutter/material.dart';
import 'package:flutter_localizations/flutter_localizations.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';

import 'package:flutter_gen/gen_l10n/app_localizations.dart';
import 'router.dart';                       // your GoRouter tree
import 'ui/auth_gate.dart';
import 'shared/providers/auth_provider.dart';
import 'shared/providers/locale_notifier.dart';   // step 6

import 'package:amplify_flutter/amplify_flutter.dart';
import 'package:amplify_auth_cognito/amplify_auth_cognito.dart';
import 'package:amplify_storage_s3/amplify_storage_s3.dart';
import 'amplifyconfiguration.dart';

/// Riverpod provider that holds (and persists) the current locale

Future<void> _configureAmplify(WidgetRef ref) async {
  if (Amplify.isConfigured) return;

  await Amplify.addPlugins([
    AmplifyAuthCognito(),
    AmplifyStorageS3(),
  ]);
  await Amplify.configure(amplifyconfig);

  // check session once Amplify is ready
  ref.read(authProvider.notifier).checkLoginStatus();
}

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const ProviderScope(child: _Bootstrapper()));
}

/// A tiny wrapper that waits for Amplify before showing the main app.
class _Bootstrapper extends ConsumerStatefulWidget {
  const _Bootstrapper({super.key});
  @override
  ConsumerState<_Bootstrapper> createState() => _BootstrapperState();
}

class _BootstrapperState extends ConsumerState<_Bootstrapper> {
  bool _ampReady = false;

  @override
  void initState() {
    super.initState();
    _configureAmplify(ref).then((_) {
      if (mounted) setState(() => _ampReady = true);
    });
  }

  @override
  Widget build(BuildContext context) {
    final locale = ref.watch(localeNotifierProvider).locale;

    final baseTheme = ThemeData(
      useMaterial3: true,
      colorSchemeSeed: const Color(0xFF23E0A7),
      textTheme: GoogleFonts.poppinsTextTheme(),
    );

    final app = MaterialApp.router(
      debugShowCheckedModeBanner: false,
      routerConfig: createRouter(),             // your existing GoRouter factory
      theme: baseTheme,
      locale: locale,
      supportedLocales: AppLocalizations.supportedLocales,
      localizationsDelegates: const [
        AppLocalizations.delegate,
        GlobalMaterialLocalizations.delegate,
        GlobalWidgetsLocalizations.delegate,
        GlobalCupertinoLocalizations.delegate,
      ],
    );

    // While Amplify is starting up, show a simple loading shell.
    return _ampReady
        ? app
        : MaterialApp(
            debugShowCheckedModeBanner: false,
            theme: baseTheme,
            locale: locale,
            supportedLocales: AppLocalizations.supportedLocales,
            localizationsDelegates: const [
              AppLocalizations.delegate,
              GlobalMaterialLocalizations.delegate,
              GlobalWidgetsLocalizations.delegate,
              GlobalCupertinoLocalizations.delegate,
            ],
            home: const Scaffold(
              body: Center(child: CircularProgressIndicator()),
            ),
          );
  }
}