import 'package:flutter/material.dart';

class NotificationsScreen extends StatefulWidget {
  const NotificationsScreen({super.key});

  @override
  State<NotificationsScreen> createState() => _NotificationsScreenState();
}

class _NotificationsScreenState extends State<NotificationsScreen> {
  bool push = true;
  bool email = false;
  bool summary = true;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Notifications')),
      body: ListView(
        children: [
          SwitchListTile(
            value: push,
            onChanged: (v) => setState(() => push = v),
            title: const Text('Push notifications'),
            subtitle: const Text('On-device alerts'),
          ),
          SwitchListTile(
            value: email,
            onChanged: (v) => setState(() => email = v),
            title: const Text('E-mail notifications'),
            subtitle: const Text('Weekly progress, marketing, etc.'),
          ),
          SwitchListTile(
            value: summary,
            onChanged: (v) => setState(() => summary = v),
            title: const Text('Daily summary'),
          ),
        ],
      ),
    );
  }
}
