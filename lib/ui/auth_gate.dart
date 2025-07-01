// lib/ui/auth_gate.dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../features/auth/login_page.dart';
import '../router.dart';
import '../shared/providers/auth_provider.dart';

/* ─────────────────────────  OPTIONAL  ─────────────────────────
   Remove once you have a real e-mail-code confirmation screen.  */
class _ConfirmCodePage extends StatelessWidget {
  const _ConfirmCodePage({required this.email});
  final String email;

  @override
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(title: const Text('Confirm e-mail')),
        body: Center(
          child: Text('Enter 6-digit code we sent to\n$email'),
        ),
      );
}
/* ───────────────────────────────────────────────────────────────*/

/// Decides whether to show the **app**, the **login screen**,
/// the **confirm-code page**, or a simple **splash / error** view.
///
/// Every branch is wrapped in **its own MaterialApp**, so widgets
/// like `Scaffold` always have `Directionality` & `Theme` ancestors.
class AuthGate extends ConsumerWidget {
  const AuthGate({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final auth = ref.watch(authProvider);

    switch (auth.status) {
      /* ─────────── already signed-in → main app ─────────── */
      case AuthStatus.authenticated:
        return MaterialApp.router(
          title: 'FitPerfect',
          debugShowCheckedModeBanner: false,
          routerConfig: router,
        );

      /* ─────────── first-time / signed-out ─────────── */
      case AuthStatus.unauthenticated:
        return MaterialApp(
          debugShowCheckedModeBanner: false,
          theme: ThemeData(useMaterial3: true),
          home: const LoginPage(),
        );

      /* ─────────── waiting for 6-digit code ─────────── */
      case AuthStatus.awaitingCode:
        return MaterialApp(
          debugShowCheckedModeBanner: false,
          theme: ThemeData(useMaterial3: true),
          home: _ConfirmCodePage(email: auth.email!),
        );

      /* ─────────── unexpected error ─────────── */
      case AuthStatus.error:
        return MaterialApp(
          debugShowCheckedModeBanner: false,
          home: Scaffold(
            body: Center(
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: Text(
                  auth.error ?? 'Unknown authentication error',
                  style: Theme.of(context).textTheme.titleMedium,
                  textAlign: TextAlign.center,
                ),
              ),
            ),
          ),
        );

      /* ─────────── app start-up / still checking ─────────── */
      case AuthStatus.unknown:
      default:
        return const MaterialApp(
          debugShowCheckedModeBanner: false,
          home: Scaffold(
            body: Center(child: CircularProgressIndicator()),
          ),
        );
    }
  }
}
