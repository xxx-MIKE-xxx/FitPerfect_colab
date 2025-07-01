import 'package:flutter/material.dart';

class EditProfileScreen extends StatefulWidget {
  const EditProfileScreen({super.key});

  @override
  State<EditProfileScreen> createState() => _EditProfileScreenState();
}

class _EditProfileScreenState extends State<EditProfileScreen> {
  final _formKey = GlobalKey<FormState>();
  String? _name;
  String? _email;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Edit profile')),
      body: Form(
        key: _formKey,
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            Center(
              child: Stack(
                alignment: Alignment.bottomRight,
                children: [
                  const CircleAvatar(radius: 48, child: Icon(Icons.person, size: 48)),
                  FloatingActionButton.small(
                    heroTag: 'avatar',
                    onPressed: () {},
                    child: const Icon(Icons.photo_camera),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 32),
            TextFormField(
              decoration: const InputDecoration(labelText: 'Name'),
              onSaved: (v) => _name = v,
              validator: (v) => v == null || v.isEmpty ? 'Required' : null,
            ),
            const SizedBox(height: 16),
            TextFormField(
              decoration: const InputDecoration(labelText: 'E-mail'),
              onSaved: (v) => _email = v,
              validator: (v) => v == null || !v.contains('@') ? 'Enter a valid e-mail' : null,
            ),
            const SizedBox(height: 32),
            FilledButton(
              onPressed: () {
                if (_formKey.currentState?.validate() ?? false) {
                  _formKey.currentState?.save();
                  // TODO: call API
                  ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Saved')));
                }
              },
              child: const Text('Save changes'),
            ),
          ],
        ),
      ),
    );
  }
}
