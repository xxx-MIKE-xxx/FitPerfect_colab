// lib/shared/providers/auth_provider.dart
import 'dart:async';
import 'dart:developer' as dev;

import 'package:amplify_auth_cognito/amplify_auth_cognito.dart'
    show CognitoSignInWithWebUIPluginOptions;
import 'package:amplify_flutter/amplify_flutter.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

enum AuthStatus { unknown, unauthenticated, awaitingCode, authenticated, error }

const bool _enableAuthLogs = false;

void _authLog(
  String message, {
  Object? error,
  StackTrace? stackTrace,
}) {
  if (!_enableAuthLogs) return;
  dev.log(message, error: error, stackTrace: stackTrace);
}

class AuthState {
  final AuthStatus status;
  final String? email;
  final String? error;
  const AuthState(this.status, {this.email, this.error});
  const AuthState.unknown()         : this(AuthStatus.unknown);
  const AuthState.unauthenticated() : this(AuthStatus.unauthenticated);
  const AuthState.authenticated()   : this(AuthStatus.authenticated);
  const AuthState.awaitingCode(String email) : this(AuthStatus.awaitingCode, email: email);
  const AuthState.error(String msg) : this(AuthStatus.error, error: msg);
}

class AuthNotifier extends StateNotifier<AuthState> {
  AuthNotifier() : super(const AuthState.unknown()) {
    _hubSub = Amplify.Hub.listen(HubChannel.Auth, (e) {
      _authLog('Hub[Auth]: ${e.eventName}  payload=${e.payload}');
    });
    _checkLoginStatus();
  }

  StreamSubscription? _hubSub;
  bool _webUiInFlight = false;

  @override
  void dispose() {
    _hubSub?.cancel();
    super.dispose();
  }

  Future<void> checkLoginStatus() => _checkLoginStatus();

  Future<void> _checkLoginStatus() async {
    try {
      _authLog('Auth: fetchAuthSession…');
      final session = await Amplify.Auth.fetchAuthSession();
      _authLog('Auth: isSignedIn=${session.isSignedIn}');
      state = session.isSignedIn ? const AuthState.authenticated()
                                 : const AuthState.unauthenticated();
    } on AuthException catch (e, st) {
      _authLog('Auth: fetchAuthSession ERROR: ${e.message}', stackTrace: st);
      state = AuthState.error(e.message);
    }
  }

  Future<void> signUp({required String email, required String password}) async {
    try {
      _authLog('Auth: signUp($email)…');
      await Amplify.Auth.signUp(
        username: email,
        password: password,
        options: SignUpOptions(userAttributes: {AuthUserAttributeKey.email: email}),
      );
      state = AuthState.awaitingCode(email);
    } on AuthException catch (e, st) {
      _authLog('Auth: signUp ERROR: ${e.message}', stackTrace: st);
      state = AuthState.error(e.message);
    }
  }

  Future<void> confirmCode({required String email, required String code}) async {
    try {
      _authLog('Auth: confirmSignUp($email)…');
      final res = await Amplify.Auth.confirmSignUp(username: email, confirmationCode: code);
      _authLog('Auth: confirmSignUp → isSignUpComplete=${res.isSignUpComplete}');
      if (res.isSignUpComplete) {
        await signIn(email: email, password: null);
      } else {
        state = const AuthState.unauthenticated();
      }
    } on AuthException catch (e, st) {
      _authLog('Auth: confirmSignUp ERROR: ${e.message}', stackTrace: st);
      state = AuthState.error(e.message);
    }
  }

  Future<void> signIn({required String email, required String? password}) async {
    try {
      _authLog('Auth: signIn($email)…');
      await Amplify.Auth.signIn(username: email, password: password);
      await _checkLoginStatus();
    } on AuthException catch (e, st) {
      _authLog('Auth: signIn ERROR: ${e.message}', stackTrace: st);
      state = AuthState.error(e.message);
    }
  }

  Future<void> signOut() async {
    try {
      _authLog('Auth: signOut(global)…');
      await Amplify.Auth.signOut(options: const SignOutOptions(globalSignOut: true));
      await _checkLoginStatus();
    } on AuthException catch (e, st) {
      _authLog('Auth: signOut ERROR: ${e.message}', stackTrace: st);
      state = AuthState.error(e.message);
    }
  }

  /// Google / Facebook via Hosted UI
  Future<void> signInWithProvider(AuthProvider provider) async {
    if (_webUiInFlight) {
      _authLog('HostedUI: sign-in ignored (already in flight)');
      return;
    }
    _webUiInFlight = true;
    _authLog('HostedUI: start → $provider');
    try {
      final opts = SignInWithWebUIOptions(
        pluginOptions: const CognitoSignInWithWebUIPluginOptions(
          isPreferPrivateSession: false, // iOS: must be non-ephemeral
        ),
      );
      final res = await Amplify.Auth.signInWithWebUI(provider: provider, options: opts);
      _authLog('HostedUI: result → nextStep=${res.nextStep.signInStep} isSignedIn=${res.isSignedIn}');
      await _checkLoginStatus();
    } on AuthException catch (e, st) {
      _authLog('HostedUI: ERROR: ${e.runtimeType}: ${e.message}', stackTrace: st);
      if (e.message.toLowerCase().contains('cancel')) {
        _authLog('Hint: iOS reported cancel. Check Info.plist URL scheme, '
            'Cognito callback (com.fitperfect://auth), and ensure only one session is launched.');
      }
      state = AuthState.error(e.message);
    } finally {
      _webUiInFlight = false;
    }
  }

  /// Full Hosted UI (lets user choose provider on the Cognito page)
  Future<void> signInHostedUI() async {
    if (_webUiInFlight) return;
    _webUiInFlight = true;
    _authLog('HostedUI (no provider): start');
    try {
      final opts = SignInWithWebUIOptions(
        pluginOptions: const CognitoSignInWithWebUIPluginOptions(
          isPreferPrivateSession: false,
        ),
      );
      final res = await Amplify.Auth.signInWithWebUI(options: opts);
      _authLog('HostedUI (no provider): result → nextStep=${res.nextStep.signInStep} '
          'isSignedIn=${res.isSignedIn}');
      await _checkLoginStatus();
    } on AuthException catch (e, st) {
      _authLog('HostedUI (no provider): ERROR: ${e.message}', stackTrace: st);
      state = AuthState.error(e.message);
    } finally {
      _webUiInFlight = false;
    }
  }

  /// Prints a ready-to-open Hosted UI URL (paste in Safari on the device)
  void printHostedUiTestUrl({AuthProvider provider = AuthProvider.google}) {
    const domain =
        'https://fitperfect69642e95-69642e95-dev.auth.eu-central-1.amazoncognito.com';
    const clientId = 'lfv9iu16cn26scobn3k6h2mh8';
    const redirect = 'com.fitperfect://auth';
    final scope = 'openid+email+profile';
    final url =
        '$domain/oauth2/authorize?identity_provider=${provider.name}'
        '&redirect_uri=${Uri.encodeComponent(redirect)}'
        '&response_type=code&client_id=$clientId&scope=$scope';
    _authLog('HostedUI TEST URL → $url');
  }
}

final authProvider =
    StateNotifierProvider<AuthNotifier, AuthState>((ref) => AuthNotifier());
