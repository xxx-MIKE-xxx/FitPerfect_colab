enum AuthStatus {
  unknown,        // splash / checking tokens
  unauthenticated, // show login / sign-up
  awaitingCode,    // show confirm-code form
  authenticated,
  error,
}

class AuthState {
  final AuthStatus status;
  final String? email;          // keep the address between steps
  final String? error;

  const AuthState(
    this.status, {
    this.email,
    this.error,
  });

  const AuthState.unknown()           : this(AuthStatus.unknown);
  const AuthState.unauthenticated()   : this(AuthStatus.unauthenticated);
  const AuthState.authenticated()     : this(AuthStatus.authenticated);
  const AuthState.awaitingCode(String email)
      : this(AuthStatus.awaitingCode, email: email);
  const AuthState.error(String msg)
      : this(AuthStatus.error, error: msg);
}
