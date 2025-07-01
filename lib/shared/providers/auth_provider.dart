// lib/shared/providers/auth_provider.dart
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:amplify_flutter/amplify_flutter.dart';

/* ───────────────────── state model ───────────────────── */

enum AuthStatus {
  unknown,          // splash / still checking
  unauthenticated,  // show login / sign-up page
  awaitingCode,     // waiting for 6-digit confirmation code
  authenticated,
  error,            // holds an error message
}

class AuthState {
  final AuthStatus status;
  final String? email;   // keep e-mail between sign-up ► confirm
  final String? error;   // human-readable error message

  const AuthState(this.status, {this.email, this.error});

  const AuthState.unknown()         : this(AuthStatus.unknown);
  const AuthState.unauthenticated() : this(AuthStatus.unauthenticated);
  const AuthState.authenticated()   : this(AuthStatus.authenticated);
  const AuthState.awaitingCode(String email)
      : this(AuthStatus.awaitingCode, email: email);
  const AuthState.error(String msg) : this(AuthStatus.error, error: msg);
}

/* ───────────────────── notifier ───────────────────── */

class AuthNotifier extends StateNotifier<AuthState> {
  AuthNotifier() : super(const AuthState.unknown()) {
    _checkLoginStatus();                // run once at start-up
  }

  /* ---------- helper main.dart calls again ---------- */
  Future<void> checkLoginStatus() => _checkLoginStatus();

  Future<void> _checkLoginStatus() async {
    try {
      final session = await Amplify.Auth.fetchAuthSession();
      state = session.isSignedIn
          ? const AuthState.authenticated()
          : const AuthState.unauthenticated();
    } on AuthException catch (e) {
      state = AuthState.error(e.message);
    }
  }

  /* ---------- SIGN-UP ---------- */

  Future<void> signUp({
    required String email,
    required String password,
  }) async {
    try {
      await Amplify.Auth.signUp(
        username: email,
        password: password,
        options: SignUpOptions(
          userAttributes: {AuthUserAttributeKey.email: email},
        ),
      );
      state = AuthState.awaitingCode(email);
    } on AuthException catch (e) {
      state = AuthState.error(e.message);
    }
  }

  Future<void> confirmCode({
    required String email,
    required String code,
  }) async {
    try {
      final res = await Amplify.Auth.confirmSignUp(
        username: email,
        confirmationCode: code,
      );

      if (res.isSignUpComplete) {
        // auto-login after a successful confirmation
        await signIn(email: email, password: null);
      } else {
        state = const AuthState.unauthenticated();
      }
    } on AuthException catch (e) {
      state = AuthState.error(e.message);
    }
  }

  /* ---------- SIGN-IN / SIGN-OUT ---------- */

  Future<void> signIn({
    required String email,
    required String? password,
  }) async {
    try {
      await Amplify.Auth.signIn(username: email, password: password);
      state = const AuthState.authenticated();
    } on AuthException catch (e) {
      state = AuthState.error(e.message);
    }
  }

  /// Full sign-out. `globalSignOut: true` already logs out from all
  /// devices *and* clears the local key-chain cache on Flutter.
  Future<void> signOut() async {
    try {
      await Amplify.Auth.signOut(
        options: const SignOutOptions(globalSignOut: true),
      );
      await _checkLoginStatus();             // refresh provider state
    } on AuthException catch (e) {
      state = AuthState.error(e.message);
    }
  }

  /* ---------- SOCIAL (Google / Facebook) ---------- */

  Future<void> signInWithProvider(AuthProvider provider) async {
    try {
      await Amplify.Auth.signInWithWebUI(provider: provider);
      state = const AuthState.authenticated();
    } on AuthException catch (e) {
      state = AuthState.error(e.message);
    }
  }
}

/* ─────────────────── provider singleton ─────────────────── */

final authProvider =
    StateNotifierProvider<AuthNotifier, AuthState>((ref) => AuthNotifier());
