import 'package:flutter/material.dart';

class SaveAutoDeleteScreen extends StatefulWidget {
  const SaveAutoDeleteScreen({super.key});

  @override
  State<SaveAutoDeleteScreen> createState() => _SaveAutoDeleteScreenState();
}

class _SaveAutoDeleteScreenState extends State<SaveAutoDeleteScreen> {
  bool autoDelete = true;
  double days = 30;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Save & auto-delete')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          SwitchListTile(
            value: autoDelete,
            onChanged: (v) => setState(() => autoDelete = v),
            title: const Text('Enable auto-delete'),
          ),
          if (autoDelete) ...[
            const SizedBox(height: 16),
            Text('Delete after ${days.round()} days', style: Theme.of(context).textTheme.bodyLarge),
            Slider(
              value: days,
              onChanged: (v) => setState(() => days = v),
              min: 1,
              max: 90,
              divisions: 89,
            ),
          ],
        ],
      ),
    );
  }
}
