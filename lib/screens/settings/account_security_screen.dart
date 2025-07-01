import 'package:flutter/material.dart';

class AccountSecurityScreen extends StatelessWidget {
  const AccountSecurityScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Account & security')),
      body: ListView(
        children: [
          ListTile(
            leading: const Icon(Icons.password),
            title: const Text('Change password'),
            onTap: () {},
          ),
          ListTile(
            leading: const Icon(Icons.phonelink_lock),
            title: const Text('Two-factor authentication'),
            subtitle: const Text('SMS · Authenticator app'),
            onTap: () {},
          ),
          ListTile(
            leading: const Icon(Icons.devices),
            title: const Text('Manage devices'),
            onTap: () {},
          ),
        ],
      ),
    );
  }
}
