import 'package:flutter/material.dart';

class ExportDataScreen extends StatelessWidget {
  const ExportDataScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Export data')),
      body: Center(
        child: FilledButton.icon(
          icon: const Icon(Icons.download),
          label: const Text('Download ZIP'),
          onPressed: () {}, // TODO: implement export
        ),
      ),
    );
  }
}
