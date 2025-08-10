// lib/screens/exercise_preview_screen.dart
// ─────────────────────────────────────────────────────────────────────────────
import 'dart:async';
import 'dart:convert';
import 'dart:io';

import '../shared/services/api_client.dart';
import 'package:amplify_flutter/amplify_flutter.dart' as amp;
import 'package:amplify_auth_cognito/amplify_auth_cognito.dart';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:go_router/go_router.dart';
import 'package:http/http.dart' as http;
import 'package:intl/intl.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';            // ← local save
import '../shared/services/s3_uploader.dart';
import '../shared/services/video_transcoder.dart';            // ← NEW
// ─────────────────────────────────────────────────────────────────────────────

class ExercisePreviewScreen extends StatefulWidget {
  const ExercisePreviewScreen({super.key, required this.exerciseId});
  final String exerciseId;

  @override
  State<ExercisePreviewScreen> createState() => _ExercisePreviewScreenState();
}

class _ExercisePreviewScreenState extends State<ExercisePreviewScreen> {
  late final CameraController _cam;
  bool _ready     = false;
  bool _recording = false;
  final _picker   = ImagePicker();

  /* ───────────── progress helper ───────────── */
  Future<void> _showProcessingDialog(VoidCallback onSkip) async {
    debugPrint('[ExercisePreview] showing “Analyzing…” dialog');
    await showDialog(
      context: context,
      barrierDismissible: false,
      useRootNavigator: false,
      builder: (_) => AlertDialog(
        content: const Row(
          children: [
            SizedBox(
              width: 28,
              height: 28,
              child: CircularProgressIndicator(strokeWidth: 3),
            ),
            SizedBox(width: 24),
            Expanded(child: Text('Analyzing your video…')),
          ],
        ),
        actions: [
          TextButton(
            onPressed: onSkip,
            child: const Text('Show replay'),
          ),
        ],
      ),
    );
  }

  void _hideProcessingDialog() {
    while (Navigator.of(context).canPop()) {
      Navigator.of(context).pop();
    }
    debugPrint('[ExercisePreview] processing dialog closed');
  }
  // ────────────────────────────────────────────

  Future<File?> pickVideoFromGallery() async {
    final xFile = await _picker.pickVideo(source: ImageSource.gallery);
    return xFile == null ? null : File(xFile.path);
  }

  /* ───────────── UPLOAD & PROCESS ───────────── */
  Future<void> _uploadAndProcess(File videoFile) async {
    final ts     = DateFormat('yyyy-MM-dd_HH-mm-ss').format(DateTime.now());
    final s3Path = 'private/videos/${widget.exerciseId}/$ts.mp4';

    /* A. 💾 persist local copy ──────────────────────────────────────── */
    String localPath = videoFile.path;
    try {
      final dir   = await getApplicationDocumentsDirectory();
      final name  = s3Path.replaceAll('/', '_');
      final saved = await videoFile.copy('${dir.path}/$name');
      localPath   = saved.path;
      debugPrint('[ExercisePreview] local copy → $localPath');
    } catch (e) {
      debugPrint('[ExercisePreview] could not save video locally: $e');
    }

    /* NEW: 🔄 make 10-fps surrogate ────────────────────────────────── */
    late File uploadFile;
    try {
      uploadFile = await VideoTranscoder.to10Fps(videoFile);
      debugPrint('[ExercisePreview] 10-fps copy → ${uploadFile.path}');
    } catch (e) {
      debugPrint('[ExercisePreview] transcode failed → $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Could not convert video to 10 fps')),
        );
      }
      return;
    }

    /* B. ⬆️  S3 upload ─────────────────────────────────────────────── */
    final presignedUrl = await S3Uploader.upload(uploadFile, s3Path); // ← uploadFile
    debugPrint('[ExercisePreview] S3 upload done');

    /* C. ☁️  notify backend ───────────────────────────────────────── */
    await _notifyBackend(s3Path);

    /* D. 🔄  polling for report ───────────────────────────────────── */
    try {
      final skipCompleter = Completer<Map<String, dynamic>>();
      _showProcessingDialog(() {
        _hideProcessingDialog();
        if (!skipCompleter.isCompleted) skipCompleter.complete({});
      });

      final report = await Future.any([
        ApiClient.fetchReport(
          s3Path,
          delay: const Duration(seconds: 6),
          max: 120,
        ),
        skipCompleter.future,
      ]);

      _hideProcessingDialog();

      if (skipCompleter.isCompleted) {
        debugPrint('[ExercisePreview] bypassing processing – showing replay');
        _navigateToFeedback(localPath, s3Path, {});
      } else {
        debugPrint('✅ report downloaded for $s3Path');
        _navigateToFeedback(localPath, s3Path, report);
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('🎉 Analysis finished')),
        );
      }
    } catch (e) {
      _hideProcessingDialog();
      debugPrint('[ExercisePreview] fetchReport failed → $e');
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Could not fetch report – $e')),
      );
    }

    // optional toast with presigned URL
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('📡 Uploaded!\n$s3Path'),
          action : SnackBarAction(
            label: 'Open',
            onPressed: () => launchUrl(Uri.parse(presignedUrl)),
          ),
        ),
      );
    }
  }

  /* ───────────── NAVIGATION (→ feedback) ───────────── */
  void _navigateToFeedback(
    String videoPath,
    String s3Path,
    Map<String, dynamic> report,
  ) {
    if (!mounted) return;

    debugPrint('[ExercisePreview] navigating to /feedback …');
    try {
      context.go(
        '/feedback',
        extra: {
          'videoPath': videoPath,
          'videoKey' : s3Path,
          'report'   : report,
        },
      );
    } catch (e) {
      debugPrint('[ExercisePreview] navigation error: $e');
    }
  }

  /* ─────────────────── CAMERA ─────────────────── */
  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    final cams = await availableCameras();
    final rear = cams.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.back,
      orElse: () => cams.first,
    );

    _cam = CameraController(rear, ResolutionPreset.high, enableAudio: true);
    await _cam.initialize();
    await _cam.lockCaptureOrientation(DeviceOrientation.portraitUp);

    if (mounted) setState(() => _ready = true);
  }

  @override
  void dispose() {
    _cam.dispose();
    super.dispose();
  }

  /* ───────────────────── UI ───────────────────── */
  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: Text(widget.exerciseId.capitalize()),
        leading: const BackButton(),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 16),
            child: Icon(_iconFor(widget.exerciseId)),
          ),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: _ready
                ? FittedBox(
                    fit: BoxFit.cover,
                    child: SizedBox(
                      width : _cam.value.previewSize!.height,
                      height: _cam.value.previewSize!.width,
                      child : CameraPreview(_cam),
                    ),
                  )
                : const Center(child: CircularProgressIndicator()),
          ),
          // ─ buttons ────────────────────────────────────────────
          Padding(
            padding: const EdgeInsets.fromLTRB(32, 24, 32, 40),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                FilledButton(
                  style: FilledButton.styleFrom(
                    minimumSize: const Size.fromHeight(54),
                    backgroundColor:
                        _recording ? Colors.red : theme.colorScheme.primary,
                  ),
                  onPressed: _ready ? _toggleRecording : null,
                  child: Text(_recording ? 'Stop recording'
                                        : 'Start recording'),
                ),
                const SizedBox(height: 12),
                FilledButton.tonalIcon(
                  icon : const Icon(Icons.video_library),
                  label: const Text('Choose existing video'),
                  onPressed: () async {
                    final file = await pickVideoFromGallery();
                    if (file == null) return;
                    if (!await _ensureSignedIn()) return;
                    await _uploadAndProcess(file);
                  },
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  /* ─────────────────── HELPERS ─────────────────── */
  Future<bool> _ensureSignedIn() async {
    try {
      if ((await amp.Amplify.Auth.fetchAuthSession()).isSignedIn) return true;
    } on amp.AuthException catch (e) {
      amp.safePrint('‼︎ Auth check failed → $e');
    }

    if (!mounted) return false;

    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Please sign-in to upload workouts')),
    );

    await context.push('/login');
    try {
      return (await amp.Amplify.Auth.fetchAuthSession()).isSignedIn;
    } catch (_) {
      return false;
    }
  }

  Future<void> _toggleRecording() async {
    if (!await _ensureSignedIn()) return;

    if (_recording) {
      final xFile = await _cam.stopVideoRecording();
      setState(() => _recording = false);
      await _uploadAndProcess(File(xFile.path));
    } else {
      await _cam.prepareForVideoRecording();
      await _cam.startVideoRecording();
      setState(() => _recording = true);
    }
  }

  // POST { "s3_key": "<path>", … } to Flask /enqueue
  Future<void> _notifyBackend(String s3Path) async {
    const backendEndpoint = 'http://63.178.80.242:5001/enqueue';

    try {
      final sess = await amp.Amplify.Auth.fetchAuthSession()
          as CognitoAuthSession;
      final jwt  = sess.userPoolTokensResult.valueOrNull?.accessToken.raw;

      final resp = await http.post(
        Uri.parse(backendEndpoint),
        headers: {
          'Content-Type': 'application/json',
          if (jwt != null) 'Authorization': 'Bearer $jwt',
        },
        body: jsonEncode({
          's3_key'     : s3Path,
          'exercise_id': widget.exerciseId,
        }),
      );

      resp.statusCode == 202
          ? debugPrint('✅ backend accepted enqueue')
          : debugPrint('⚠️ backend ${resp.statusCode}: ${resp.body}');
    } catch (e) {
      debugPrint('⚠️ could not reach backend: $e');
    }
  }

  IconData _iconFor(String id) => switch (id) {
        'squat'    => Icons.fitness_center,
        'deadlift' => Icons.accessibility_new,
        _          => Icons.sports_gymnastics,
      };
}

/* ───────── util ───────── */
extension on String {
  String capitalize() =>
      isEmpty ? this : '${this[0].toUpperCase()}${substring(1)}';
}
