// ───────────────────── lib/features/auth/login_page.dart ─────────────────────
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:amplify_flutter/amplify_flutter.dart';

import '../../shared/providers/auth_provider.dart';

/* ───────── reusable social-login button ───────── */

class _SocialButton extends StatelessWidget {
  const _SocialButton({
    required this.text,
    required this.color,
    this.onTap,
    this.textColor,
  });

  final String text;
  final Color color;
  final Color? textColor;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) => SizedBox(
        width: double.infinity,
        child: ElevatedButton(
          style: ElevatedButton.styleFrom(
            backgroundColor: color,
            foregroundColor: textColor ?? Colors.white,
            padding: const EdgeInsets.symmetric(vertical: 14),
            shape:
                RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
          ),
          onPressed: onTap,
          child: Text(text),
        ),
      );
}

/* ───────── root widget decides which form to show ───────── */

class LoginPage extends ConsumerWidget {
  const LoginPage({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final auth = ref.watch(authProvider);

    switch (auth.status) {
      case AuthStatus.unauthenticated:
      case AuthStatus.error:
        return _EmailPasswordForm(error: auth.error);

      case AuthStatus.awaitingCode:
        return _ConfirmCodeForm(email: auth.email!);

      default: // unknown / authenticated handled elsewhere
        return const Scaffold(
          body: Center(child: CircularProgressIndicator()),
        );
    }
  }
}

/* ───────── email + password sign-in / sign-up ───────── */

class _EmailPasswordForm extends ConsumerStatefulWidget {
  const _EmailPasswordForm({this.error});
  final String? error;

  @override
  ConsumerState<_EmailPasswordForm> createState() =>
      _EmailPasswordFormState();
}

class _EmailPasswordFormState extends ConsumerState<_EmailPasswordForm> {
  final _form = GlobalKey<FormState>();
  final _email = TextEditingController();
  final _pwd   = TextEditingController();
  bool _busy = false;
  bool _signUpMode = false;

  Future<void> _signInWith(AuthProvider provider) async {
    setState(() => _busy = true);
    await ref.read(authProvider.notifier).signInWithProvider(provider);
    if (mounted) setState(() => _busy = false);
  }

  Future<void> _submit() async {
    if (!_form.currentState!.validate()) return;

    setState(() => _busy = true);
    final notifier = ref.read(authProvider.notifier);
    if (_signUpMode) {
      await notifier.signUp(
        email: _email.text.trim(),
        password: _pwd.text,
      );
    } else {
      await notifier.signIn(
        email: _email.text.trim(),
        password: _pwd.text,
      );
    }
    if (mounted) setState(() => _busy = false);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 340),
          child: Form(
            key: _form,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                TextFormField(
                  controller: _email,
                  decoration: const InputDecoration(labelText: 'E-mail'),
                  validator: (v) =>
                      (v != null && v.contains('@')) ? null : 'Enter e-mail',
                ),
                const SizedBox(height: 12),
                TextFormField(
                  controller: _pwd,
                  obscureText: true,
                  decoration: const InputDecoration(labelText: 'Password'),
                  validator: (v) =>
                      (v != null && v.length >= 6) ? null : 'Min 6 chars',
                ),

                const SizedBox(height: 24),

                // ── social login buttons ──
                _SocialButton(
                  text: 'Continue with Google',
                  color: Colors.white,
                  textColor: Colors.black87,
                  onTap: _busy ? null : () => _signInWith(AuthProvider.google),
                ),
                const SizedBox(height: 12),
                _SocialButton(
                  text: 'Continue with Facebook',
                  color: const Color(0xFF1877F2),
                  onTap: _busy ? null : () => _signInWith(AuthProvider.facebook),
                ),

                const SizedBox(height: 24),

                if (widget.error != null) ...[
                  Text(widget.error!, style: const TextStyle(color: Colors.red)),
                  const SizedBox(height: 12),
                ],

                FilledButton(
                  onPressed: _busy ? null : _submit,
                  child: _busy
                      ? const SizedBox.square(
                          dimension: 16,
                          child: CircularProgressIndicator(strokeWidth: 2))
                      : Text(_signUpMode ? 'Sign up' : 'Sign in'),
                ),
                TextButton(
                  onPressed: () =>
                      setState(() => _signUpMode = !_signUpMode),
                  child: Text(_signUpMode
                      ? 'Have an account? Sign in'
                      : 'New here? Sign up'),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

/* ───────── confirmation-code form ───────── */

class _ConfirmCodeForm extends ConsumerStatefulWidget {
  const _ConfirmCodeForm({required this.email});
  final String email;

  @override
  ConsumerState<_ConfirmCodeForm> createState() =>
      _ConfirmCodeFormState();
}

class _ConfirmCodeFormState extends ConsumerState<_ConfirmCodeForm> {
  final _form = GlobalKey<FormState>();
  final _code = TextEditingController();
  bool _busy = false;
  String? _error;

  Future<void> _submit() async {
    if (!_form.currentState!.validate()) return;
    setState(() {
      _busy = true;
      _error = null;
    });
    await ref.read(authProvider.notifier).confirmCode(
          email: widget.email,
          code: _code.text.trim(),
        );
    if (mounted) setState(() => _busy = false);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 300),
          child: Form(
            key: _form,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text('We sent a 6-digit code to\n${widget.email}',
                    textAlign: TextAlign.center),
                const SizedBox(height: 16),
                TextFormField(
                  controller: _code,
                  textAlign: TextAlign.center,
                  decoration: const InputDecoration(
                      labelText: 'Confirmation code'),
                  validator: (v) =>
                      (v != null && v.length == 6) ? null : 'Enter 6 digits',
                ),
                if (_error != null) ...[
                  const SizedBox(height: 12),
                  Text(_error!, style: const TextStyle(color: Colors.red)),
                ],
                const SizedBox(height: 24),
                FilledButton(
                  onPressed: _busy ? null : _submit,
                  child: _busy
                      ? const SizedBox.square(
                          dimension: 16,
                          child: CircularProgressIndicator(strokeWidth: 2))
                      : const Text('Confirm'),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
