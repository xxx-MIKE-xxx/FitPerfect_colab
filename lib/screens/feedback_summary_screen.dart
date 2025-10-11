// lib/screens/feedback_summary_screen.dart
//
// Purpose:
// - Summarize MotionBERT 3D output after the recording pipeline.
// - If `out3dPath != null`, load file and display simple metrics.
// - If null, show retry hints.
//
// Navigation expectation (example):
//   Navigator.of(context).pushNamed(
//     '/feedback-summary',
//     arguments: {
//       'exerciseId': exerciseId,
//       'sessionId' : sessionId,
//       'out3dPath' : out3dPath, // may be null
//       'report'    : report,    // optional
//       's3Key'     : s3Key,     // optional
//     },
//   );
//
// This screen is intentionally read-only; it does not re-run the pipeline.
// If you want a "Retry 3D" button here, wire PoseProcessingController via Provider
// and call run3DForSession(sessionId, Size(w,h)).
//
// Author: FitPerfect pipeline integration

import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:path/path.dart' as p;
import 'package:share_plus/share_plus.dart';

import '../shared/services/storage_layout.dart';

class FeedbackSummaryScreen extends StatefulWidget {
  const FeedbackSummaryScreen({
    super.key,
    required this.exerciseId,
    this.sessionId,
    this.out3dPath,
    this.report,
    this.s3Key,
  });

  final String exerciseId;
  final String? sessionId;
  final String? out3dPath;
  final Map<String, dynamic>? report;
  final String? s3Key;

  @override
  State<FeedbackSummaryScreen> createState() => _FeedbackSummaryScreenState();
}

class _FeedbackSummaryScreenState extends State<FeedbackSummaryScreen> {
  bool _loading = false;
  String? _error;
  Map<String, dynamic>? _summary; // parsed 3D summary
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
        debugPrint('[FeedbackSummary] Ignoring non-JSON out3dPath: $raw');
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
      Map<String, dynamic> root;
      try {
        root = jsonDecode(text) as Map<String, dynamic>;
      } catch (_) {
        // If JSONL or non-map JSON, just surface basic info.
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
        debugPrint('[FeedbackSummary] Failed to load 3D: $e\n$st');
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
    // Basic schema heuristics mirroring MotionBertRunner variations.
    int frames = 0;
    int joints = 0;
    String shape = '';
    bool pelvisCentered = false;
    int windowSize = 0;
    int stride = 0;

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

    // Extra meta if present
    if (root.containsKey('windowSize')) {
      final v = root['windowSize'];
      if (v is int) windowSize = v;
    }
    if (root.containsKey('stride')) {
      final v = root['stride'];
      if (v is int) stride = v;
    }
    if (root.containsKey('pelvisCentered')) {
      final v = root['pelvisCentered'];
      if (v is bool) pelvisCentered = v;
    }

    return {
      'frames': frames,
      'joints': joints,
      'shape': shape,
      'windowSize': windowSize,
      'stride': stride,
      'pelvisCentered': pelvisCentered,
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
        title: const Text('Feedback Summary'),
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
        child: has3d ? _build3D(context) : _buildNo3D(context),
      ),
    );
  }

  Widget _buildNo3D(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'No 3D results available',
          style: Theme.of(context).textTheme.titleLarge,
        ),
        const SizedBox(height: 8),
        Text(
          'We did not receive a path to the 3D output. '
          'Return to the previous screen and rerun the 3D step after stopping the stream.',
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
              label: const Text('Back'),
            ),
          ],
        ),
      ],
    );
  }

  Widget _build3D(BuildContext context) {
    final out3dPath = _out3dPath!;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          '3D results',
          style: Theme.of(context).textTheme.titleLarge,
        ),
        const SizedBox(height: 12),
        _loading
            ? const LinearProgressIndicator()
            : _error != null
                ? _errorCard(_error!)
                : _summaryCard(out3dPath, _summary),
        const SizedBox(height: 16),
        Wrap(
          spacing: 12,
          runSpacing: 12,
          children: [
            ElevatedButton.icon(
              onPressed: () {
                // Placeholder for a richer 3D viewer route.
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text('3D viewer not implemented in this screen.')),
                );
              },
              icon: const Icon(Icons.view_in_ar),
              label: const Text('Open 3D Viewer'),
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
    final tiles = <Widget>[
      ListTile(
        leading: const Icon(Icons.insert_drive_file),
        title: Text(p.basename(out3dPath)),
        subtitle: Text(p.dirname(out3dPath)),
      ),
    ];
    if (summary != null) {
      tiles.addAll([
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
        if ((summary['windowSize'] as int? ?? 0) > 0)
          ListTile(
            leading: const Icon(Icons.view_stream),
            title: const Text('Window Size'),
            trailing: Text('${summary['windowSize']}'),
          ),
        if ((summary['stride'] as int? ?? 0) > 0)
          ListTile(
            leading: const Icon(Icons.swap_horiz),
            title: const Text('Stride'),
            trailing: Text('${summary['stride']}'),
          ),
        ListTile(
          leading: const Icon(Icons.adjust),
          title: const Text('Pelvis Centered'),
          trailing: Text('${summary['pelvisCentered'] == true ? 'Yes' : 'No'}'),
        ),
      ]);
    }
    return Card(
      clipBehavior: Clip.antiAlias,
      child: Column(children: tiles),
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
      'out_2d.jsonl': StorageLayout.out2dFile(sessionId),
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
            const Text('• Ensure Documents/FitPerfect/<sessionId>/coco_2d.jsonl exists'),
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
