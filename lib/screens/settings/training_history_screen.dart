import 'package:flutter/material.dart';

class TrainingHistoryScreen extends StatelessWidget {
  const TrainingHistoryScreen({super.key});

  @override
  Widget build(BuildContext context) {
    // TODO: replace with real data / pagination
    final fake = List.generate(10, (i) => 'Session #${i + 1} · ${(30 + i) ~/ 1} min');

    return Scaffold(
      appBar: AppBar(title: const Text('Training history')),
      body: ListView.separated(
        padding: const EdgeInsets.all(16),
        itemCount: fake.length,
        itemBuilder: (_, i) => ListTile(
          leading: const Icon(Icons.fitness_center),
          title: Text(fake[i]),
          trailing: const Icon(Icons.chevron_right),
          onTap: () {}, // maybe open a detailed report
        ),
        separatorBuilder: (_, __) => const Divider(height: 1),
      ),
    );
  }
}
