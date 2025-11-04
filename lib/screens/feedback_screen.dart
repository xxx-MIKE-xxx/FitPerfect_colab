// lib/screens/feedback_screen.dart
//
// Purpose:
// - If `out3dPath != null`, load the produced 3D JSON and show simple summary metrics.
// - If null, show a Retry panel with basic logs guidance.
//
// Expected to be navigated with something like:
//   FeedbackScreen(
//     exerciseId: exerciseId,
//     sessionId : sessionId,            // optional, for context
//     out3dPath : out3d?.path,          // optional, presence toggles UI
//     report    : report,               // optional
//     s3Key     : videoKey,             // optional
//   )
//
// Notes:
// - This screen is intentionally decoupled from the pipeline; it only *reads* files.
// - If you want a 'Retry 3D' action here, wire PoseProcessingController and call
//   controller.run3DForSession(sessionId, frameSize). For now we simply show guidance.
//
// Author: FitPerfect pipeline integration

import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:path/path.dart' as p;
import 'package:share_plus/share_plus.dart';

import '../shared/services/storage_layout.dart';

class FeedbackScreen extends StatefulWidget {
  const FeedbackScreen({
    super.key,
    required this.exerciseId,
    this.sessionId,
    this.out3dPath,
    this.report,
    this.s3Key,
  });

  final String exerciseId;
  final String? sessionId;
  final String? out3dPath; // path to out_3d.json (optional)
  final Map<String, dynamic>? report;
  final String? s3Key;

  @override
  State<FeedbackScreen> createState() => _FeedbackScreenState();
}

class _FeedbackScreenState extends State<FeedbackScreen> {
  bool _loading = false;
  String? _error;
  Map<String, dynamic>? _summary; // parsed summary from 3D json
  String? _out3dPath;

  @override
  void initState() {
    super.initState();
    _out3dPath = _sanitizeOut3dPath(widget.out3dPath);
    if (_out3dPath != null) {
      _load3d(_out3dPath!);
    }
  }

  String? _sanitizeOut3dPath(String? raw) {
    if (raw == null || raw.isEmpty) {
      return null;
    }
    final lower = raw.toLowerCase();
    const allowed = ['.json', '.jsonl'];
    final isAllowed = allowed.any(lower.endsWith);
    if (!isAllowed) {
      if (kDebugMode) {
        debugPrint('[FeedbackScreen] Ignoring non-JSON out3dPath: $raw');
      }
      return null;
    }
    return raw;
  }

  Future<void> _load3d(String path) async {
    setState(() {
      _loading = true;
      _error = null;
      _summary = null;
    });
    try {
      final f = File(path);
      if (!await f.exists()) {
        throw StateError('3D file not found at $path');
      }
      final text = await f.readAsString();
      // The JSON schema can vary depending on MotionBertRunner output.
      // We try to be robust:
      // - If { "kpts3d": [T,17,3], ... }
      // - Or { "windows": [ { "t0":..., "seq":[K x 17 x 3] }, ... ], ... }
      // - Or { "sequence": [T][17][3] }
      // We compute minimal metrics defensively.
      Map<String, dynamic> root;
      try {
        root = jsonDecode(text) as Map<String, dynamic>;
      } catch (_) {
        // If file is JSONL or array, show file info only.
        root = {
          'raw': {
            'type': 'unknown',
            'bytes': await f.length(),
            'basename': p.basename(path),
          }
        };
      }
      final summary = _summarize3d(root);
      setState(() {
        _summary = summary;
      });
    } catch (e, st) {
      if (kDebugMode) {
        debugPrint('[FeedbackScreen] Failed to load 3D: $e\n$st');
      }
      setState(() {
        _error = e.toString();
      });
    } finally {
      if (mounted) {
        setState(() {
          _loading = false;
        });
      }
    }
  }

  Future<void> _shareSessionFiles() async {
    final sessionId = widget.sessionId;
    if (sessionId == null) {
      setState(() {
        _error = 'Session ID missing – nothing to share.';
      });
      return;
    }

    try {
      final files = <XFile>[];
      final out2d = await StorageLayout.out2dFile(sessionId);
      if (await out2d.exists()) {
        files.add(XFile(out2d.path));
      }
      final out3d = await StorageLayout.out3dFile(sessionId);
      if (await out3d.exists()) {
        files.add(XFile(out3d.path));
      }
      final meta = await StorageLayout.metaFile(sessionId);
      if (await meta.exists()) {
        files.add(XFile(meta.path));
      }
      if (files.isEmpty) {
        setState(() {
          _error = 'No session files found to share.';
        });
        return;
      }
      await Share.shareXFiles(
        files,
        subject: 'FitPerfect session $sessionId',
        text: 'Session files for $sessionId',
      );
    } catch (e) {
      if (mounted) {
        setState(() {
          _error = 'Share failed: $e';
        });
      }
    }
  }

  Map<String, dynamic> _summarize3d(Map<String, dynamic> root) {
    // Heuristic summary from common keys:
    int frames = 0;
    int joints = 0;
    String shape = '';

    if (root.containsKey('kpts3d')) {
      final kpts3d = root['kpts3d'];
      if (kpts3d is List && kpts3d.isNotEmpty && kpts3d.first is List) {
        frames = kpts3d.length;
        final f0 = kpts3d.first;
        if (f0 is List) {
          joints = f0.length;
          shape = '[$frames,$joints,3]';
        }
      }
    } else if (root.containsKey('windows')) {
      final wins = root['windows'];
      if (wins is List) {
        int total = 0;
        int jts = 0;
        for (final w in wins) {
          if (w is Map && w['seq'] is List) {
            final seq = w['seq'] as List;
            total += seq.length;
            if (jts == 0 && seq.isNotEmpty && seq.first is List) {
              jts = (seq.first as List).length;
            }
          }
        }
        frames = total;
        joints = jts;
        shape = 'sum(windows)=$frames, joints=$joints';
      }
    } else if (root.containsKey('sequence')) {
      final seq = root['sequence'];
      if (seq is List && seq.isNotEmpty) {
        frames = seq.length;
        if (seq.first is List) {
          joints = (seq.first as List).length;
          shape = '[$frames,$joints,3?]';
        }
      }
    }

    return {
      'frames': frames,
      'joints': joints,
      'shape': shape,
      'has_kpts3d': root.containsKey('kpts3d'),
      'has_windows': root.containsKey('windows'),
      'has_sequence': root.containsKey('sequence'),
    };
  }

  @override
  Widget build(BuildContext context) {
    final has3d = _out3dPath != null;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Feedback'),
        actions: [
          IconButton(
            icon: const Icon(Icons.ios_share),
            tooltip: 'Share session files',
            onPressed: widget.sessionId == null ? null : _shareSessionFiles,
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: has3d ? _build3DView(context) : _buildNo3DView(context),
      ),
    );
  }

  Widget _buildNo3DView(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          '3D analysis not available',
          style: Theme.of(context).textTheme.titleLarge,
        ),
        const SizedBox(height: 8),
        Text(
          'We could not find a 3D output file for this session. '
          'You can retry the 3D step from the previous screen.',
          style: Theme.of(context).textTheme.bodyMedium,
        ),
        const SizedBox(height: 16),
        _logHelpCard(),
        const Spacer(),
        Row(
          children: [
            ElevatedButton.icon(
              onPressed: () => Navigator.of(context).pop(),
              icon: const Icon(Icons.arrow_back),
              label: const Text('Back to Preview'),
            ),
          ],
        ),
      ],
    );
  }

  Widget _build3DView(BuildContext context) {
    final summary = _summary;
    final out3dPath = _out3dPath!;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          '3D analysis ready',
          style: Theme.of(context).textTheme.titleLarge,
        ),
        const SizedBox(height: 12),
        _loading
            ? const LinearProgressIndicator()
            : _error != null
                ? _errorCard(_error!)
                : _summaryCard(out3dPath, summary),
        const SizedBox(height: 16),
        Wrap(
          spacing: 12,
          runSpacing: 12,
          children: [
            ElevatedButton.icon(
              onPressed: () {
                // Navigate to a summary/3D viewer page if you have one.
                // Replace with your route; we pass path for convenience.
                Navigator.of(context).pushNamed(
                  '/feedback-summary',
                  arguments: {
                    'exerciseId': widget.exerciseId,
                    'sessionId': widget.sessionId,
                    'out3dPath': out3dPath,
                    'report': widget.report,
                    's3Key': widget.s3Key,
                  },
                );
              },
              icon: const Icon(Icons.insights),
              label: const Text('Open Summary'),
            ),
            OutlinedButton.icon(
              onPressed: () async {
                final f = File(out3dPath);
                final sz = await f.length();
                if (mounted) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(content: Text('3D file: ${p.basename(out3dPath)}  •  ${sz}B')),
                  );
                }
              },
              icon: const Icon(Icons.description),
              label: const Text('File Info'),
            ),
          ],
        ),
        const SizedBox(height: 16),
        _buildFileInfoCard(),
      ],
    );
  }

  Widget _summaryCard(String out3dPath, Map<String, dynamic>? summary) {
    final items = <Widget>[
      ListTile(
        leading: const Icon(Icons.insert_drive_file),
        title: Text(p.basename(out3dPath)),
        subtitle: Text(p.dirname(out3dPath)),
      ),
    ];

    if (summary != null) {
      items.addAll([
        const Divider(height: 1),
        ListTile(
          leading: const Icon(Icons.timeline),
          title: const Text('Frames'),
          trailing: Text('${summary['frames'] ?? 0}'),
        ),
        ListTile(
          leading: const Icon(Icons.donut_large),
          title: const Text('Joints'),
          trailing: Text('${summary['joints'] ?? 0}'),
        ),
        if ((summary['shape'] as String?)?.isNotEmpty ?? false)
          ListTile(
            leading: const Icon(Icons.grid_on),
            title: const Text('Shape'),
            trailing: Text('${summary['shape']}'),
          ),
      ]);
    }

    return Card(
      clipBehavior: Clip.antiAlias,
      child: Column(children: items),
    );
  }

  Widget _buildFileInfoCard() {
    final sessionId = widget.sessionId;
    if (sessionId == null) {
      return const SizedBox.shrink();
    }
    return FutureBuilder<List<_SessionFileInfo>>(
      future: _collectFileInfo(sessionId),
      builder: (context, snapshot) {
        if (snapshot.connectionState != ConnectionState.done) {
          return const SizedBox.shrink();
        }
        final files = snapshot.data ?? const <_SessionFileInfo>[];
        if (files.isEmpty) {
          return const SizedBox.shrink();
        }
        return Card(
          clipBehavior: Clip.antiAlias,
          child: Column(
            children: files
                .map((info) => ListTile(
                      leading: const Icon(Icons.article_outlined),
                      title: Text(info.name),
                      subtitle: Text(p.basename(info.path)),
                      trailing: Text(
                        '${(info.size / 1024).toStringAsFixed(1)} KB\n${info.modified.toLocal().toIso8601String()}',
                        textAlign: TextAlign.end,
                        style: Theme.of(context).textTheme.bodySmall,
                      ),
                    ))
                .toList(),
          ),
        );
      },
    );
  }

  Future<List<_SessionFileInfo>> _collectFileInfo(String sessionId) async {
    final targets = <String, Future<File>>{
      'coco_2d.jsonl': StorageLayout.out2dFile(sessionId),
      'out_2d_index.json': StorageLayout.out2dIndexFile(sessionId),
      'out_3d.json': StorageLayout.out3dFile(sessionId),
      'out_3d_index.json': StorageLayout.out3dIndexFile(sessionId),
      'meta.json': StorageLayout.metaFile(sessionId),
    };
    final result = <_SessionFileInfo>[];
    for (final entry in targets.entries) {
      final file = await entry.value;
      if (await file.exists()) {
        final stat = await file.stat();
        result.add(_SessionFileInfo(
          name: entry.key,
          path: file.path,
          size: stat.size,
          modified: stat.modified,
        ));
      }
    }
    return result;
  }

  Widget _errorCard(String error) {
    return Card(
      color: Theme.of(context).colorScheme.errorContainer,
      child: Padding(
        padding: const EdgeInsets.all(12.0),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Icon(Icons.error_outline),
            const SizedBox(width: 12),
            Expanded(
              child: Text(
                error,
                style: Theme.of(context).textTheme.bodyMedium,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _logHelpCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Debug tips',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 8),
            const Text('• Ensure the app created Documents/FitPerfect/<sessionId>/coco_2d.jsonl'),
            const Text('• Check console for [2D_SAVE_OPEN]/[2D_SAVE_CLOSE] logs'),
            const Text('• Rerun 3D from the previous screen after stopping the stream'),
          ],
        ),
      ),
    );
  }
}

class _SessionFileInfo {
  _SessionFileInfo({
    required this.name,
    required this.path,
    required this.size,
    required this.modified,
  });

  final String name;
  final String path;
  final int size;
  final DateTime modified;
}
