// ─────────────────────────────────────────────────────────────────────────────
// lib/screens/exercise_preview_screen.dart
// Records video, lets you ANALYZE (offline on-device) and saves keypoints JSON.
// ─────────────────────────────────────────────────────────────────────────────
import 'dart:convert';
import 'dart:io';

import 'package:amplify_auth_cognito/amplify_auth_cognito.dart';
import 'package:amplify_flutter/amplify_flutter.dart' as amp;
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:go_router/go_router.dart';
import 'package:image_picker/image_picker.dart';
import 'package:intl/intl.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:url_launcher/url_launcher.dart';

import '../shared/services/api_client.dart';
import '../shared/services/pose_runtime.dart';      // offline pipeline
import '../shared/services/s3_uploader.dart';
import '../shared/services/video_transcoder.dart'; // 10fps helper

class ExercisePreviewScreen extends StatefulWidget {
  const ExercisePreviewScreen({super.key, required this.exerciseId});
  final String exerciseId;

  @override
  State<ExercisePreviewScreen> createState() => _ExercisePreviewScreenState();
}

class _ExercisePreviewScreenState extends State<ExercisePreviewScreen> {
  late final CameraController _cam;
  bool _ready = false;
  bool _recording = false;
  final _picker = ImagePicker();

  File? _lastVideo; // last recorded or picked video

  /* ───────────── progress helper ───────────── */
  Future<void> _showProcessingDialog() async {
    await showDialog(
      context: context,
      barrierDismissible: false,
      useRootNavigator: false,
      builder: (_) => const AlertDialog(
        content: Row(
          children: [
            SizedBox(width: 28, height: 28, child: CircularProgressIndicator(strokeWidth: 3)),
            SizedBox(width: 24),
            Expanded(child: Text('Analyzing your video…')),
          ],
        ),
      ),
    );
  }

  void _hideProcessingDialog() {
    if (!mounted) return;
    Navigator.of(context, rootNavigator: false).popUntil((route) => route is! PopupRoute);
  }
  // ────────────────────────────────────────────

  Future<File?> pickVideoFromGallery() async {
    final xFile = await _picker.pickVideo(source: ImageSource.gallery);
    return xFile == null ? null : File(xFile.path);
  }

  Future<void> _rememberVideo(File videoFile) async {
    try {
      final dir = await getApplicationDocumentsDirectory();
      final ts  = DateFormat('yyyy-MM-dd_HH-mm-ss').format(DateTime.now());
      final dst = File('${dir.path}/local_${widget.exerciseId}_$ts.mp4');
      final saved = await videoFile.copy(dst.path);
      setState(() => _lastVideo = saved);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Video ready • ${saved.path.split('/').last}')),
        );
      }
    } catch (_) {
      setState(() => _lastVideo = videoFile); // fall back
    }
  }

  /// Write a JSON report to disk. Returns the file path.
  Future<String> _saveLocalReport(Map<String, dynamic> report) async {
    final dir = await getApplicationDocumentsDirectory();
    final folder = Directory('${dir.path}/reports/${widget.exerciseId}');
    await folder.create(recursive: true);
    final ts = DateFormat('yyyy-MM-dd_HH-mm-ss').format(DateTime.now());
    final file = File('${folder.path}/$ts.json');
    await file.writeAsString(const JsonEncoder.withIndent('  ').convert(report));
    return file.path;
  }

  Future<void> _analyzeOffline() async {
    final file = _lastVideo;
    if (file == null) return;

    try {
      _showProcessingDialog();
      final pipeline = PosePipeline();
      final res = await pipeline.analyzeVideo(file);

      final report = res.toReport();                  // now contains kpts
      final savedPath = await _saveLocalReport(report);
      _hideProcessingDialog();

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('✅ Saved keypoints → ${savedPath.split('/').last}')),
        );
      }

      // Reuse your feedback flow; pass local report + local video
      _navigateToFeedback(
        file.path,
        'local/${widget.exerciseId}/${DateTime.now().millisecondsSinceEpoch}.mp4',
        report,
      );
    } catch (e) {
      _hideProcessingDialog();
      debugPrint('[ExercisePreview] offline analyze failed → $e');
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Offline analysis failed: $e')),
      );
    }
  }

  /* ───────────── UPLOAD & PROCESS (cloud fallback) ───────────── */
  Future<void> _uploadAndProcess(File videoFile) async {
    final ts     = DateFormat('yyyy-MM-dd_HH-mm-ss').format(DateTime.now());
    final s3Path = 'private/videos/${widget.exerciseId}/$ts.mp4';

    // A. local copy (for playback)
    String localPath = videoFile.path;
    try {
      final dir   = await getApplicationDocumentsDirectory();
      final name  = s3Path.replaceAll('/', '_');
      final saved = await videoFile.copy('${dir.path}/$name');
      localPath   = saved.path;
    } catch (_) {}

    // B. make 10-fps surrogate for bandwidth
    late File uploadFile;
    try {
      uploadFile = await VideoTranscoder.to10Fps(videoFile);
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Could not convert to 10 fps: $e')),
        );
      }
      return;
    }

    // C. upload
    final presignedUrl = await S3Uploader.upload(uploadFile, s3Path);

    // D. notify backend
    await _notifyBackend(s3Path);

    // E. poll for report
    try {
      _showProcessingDialog();
      final report = await ApiClient.fetchReport(
        s3Path,
        delay: const Duration(seconds: 6),
        max: 120,
      );
      _hideProcessingDialog();

      _navigateToFeedback(localPath, s3Path, report);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('🎉 Cloud analysis finished')),
        );
      }
    } catch (e) {
      _hideProcessingDialog();
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Could not fetch report – $e')),
      );
    }

    // Optional toast with URL
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
    context.go(
      '/feedback',
      extra: {
        'videoPath': videoPath,
        'videoKey' : s3Path,
        'report'   : report,
      },
    );
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

    _cam = CameraController(
      rear,
      ResolutionPreset.high,
      enableAudio: true,
      // No image streaming now, so format doesn’t matter
    );

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
      backgroundColor: const Color(0xFFF7F9FB),
      appBar: AppBar(
        elevation: 0,
        backgroundColor: const Color(0xFFF7F9FB),
        title: Row(
          children: [
            Container(
              width: 36,
              height: 36,
              decoration: const BoxDecoration(
                shape: BoxShape.circle,
                gradient: LinearGradient(
                  colors: [Color(0xFFFFC107), Color(0xFFFF7043)],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
              ),
              child: Icon(_iconFor(widget.exerciseId), color: Colors.white),
            ),
            const SizedBox(width: 12),
            Text(
              widget.exerciseId.capitalize(),
              style: theme.textTheme.titleLarge?.copyWith(
                fontWeight: FontWeight.w800,
                letterSpacing: -.2,
              ),
            ),
          ],
        ),
        leading: const BackButton(),
      ),
      body: Column(
        children: [
          // Camera preview
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

          // Action panel
          SafeArea(
            top: false,
            child: Padding(
              padding: const EdgeInsets.fromLTRB(16, 10, 16, 16),
              child: _ActionPanel(
                recording: _recording,
                hasVideo: _lastVideo != null,
                onToggleRecord: _ready ? _toggleRecording : null,
                onPickVideo: () async {
                  final file = await pickVideoFromGallery();
                  if (file == null) return;
                  await _rememberVideo(file);
                },
                onAnalyze: _analyzeOffline,
                onAnalyzeCloud: () async {
                  final f = _lastVideo;
                  if (f != null) await _uploadAndProcess(f);
                },
              ),
            ),
          ),
        ],
      ),
    );
  }

  Future<void> _toggleRecording() async {
    if (_recording) {
      final xFile = await _cam.stopVideoRecording();
      setState(() => _recording = false);
      await _rememberVideo(File(xFile.path));
      HapticFeedback.selectionClick();
    } else {
      await _cam.prepareForVideoRecording();
      await _cam.startVideoRecording();
      setState(() => _recording = true);
      HapticFeedback.heavyImpact();
    }
  }

  // POST { "s3_key": "<path>", … } to Flask /enqueue
  Future<void> _notifyBackend(String s3Path) async {
    const backendEndpoint = 'http://63.178.80.242:5001/enqueue';
    try {
      final sess = await amp.Amplify.Auth.fetchAuthSession() as CognitoAuthSession;
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

      if (resp.statusCode == 202) {
        debugPrint('✅ backend accepted enqueue');
      } else {
        debugPrint('⚠️ backend ${resp.statusCode}: ${resp.body}');
      }
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

/* ───────── UI widgets ───────── */

class _ActionPanel extends StatelessWidget {
  const _ActionPanel({
    required this.recording,
    required this.hasVideo,
    required this.onToggleRecord,
    required this.onPickVideo,
    required this.onAnalyze,
    required this.onAnalyzeCloud,
  });

  final bool recording;
  final bool hasVideo;
  final VoidCallback? onToggleRecord;
  final VoidCallback onPickVideo;
  final VoidCallback onAnalyze;
  final VoidCallback onAnalyzeCloud;

  @override
  Widget build(BuildContext context) {
    return Material(
      elevation: 10,
      color: Colors.white,
      borderRadius: BorderRadius.circular(24),
      shadowColor: Colors.black.withOpacity(.1),
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 16, 16, 16),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            _GradientButton(
              height: 56,
              onPressed: onToggleRecord,
              colors: recording
                  ? const [Color(0xFFD32F2F), Color(0xFFE53935)]
                  : const [Color(0xFFFFC107), Color(0xFFFF7043)],
              icon: recording ? Icons.stop : Icons.fiber_manual_record,
              label: recording ? 'Stop recording' : 'Tap to record',
            ),
            const SizedBox(height: 10),
            _GhostButton(
              onPressed: onPickVideo,
              icon: Icons.video_library_rounded,
              label: 'Choose existing video',
            ),
            if (hasVideo) ...[
              const SizedBox(height: 12),
              const Divider(height: 1),
              const SizedBox(height: 12),
              Row(
                children: [
                  Expanded(
                    child: _GradientButton(
                      height: 52,
                      onPressed: onAnalyze,
                      colors: const [Color(0xFF00BFA5), Color(0xFF1DE9B6)],
                      icon: Icons.analytics,
                      label: 'Analyze offline',
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: _GhostButton(
                      onPressed: onAnalyzeCloud,
                      icon: Icons.cloud_upload,
                      label: 'Analyze in cloud',
                    ),
                  ),
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _GradientButton extends StatelessWidget {
  const _GradientButton({
    required this.onPressed,
    required this.colors,
    required this.icon,
    required this.label,
    this.height = 54,
  });

  final VoidCallback? onPressed;
  final List<Color> colors;
  final IconData icon;
  final String label;
  final double height;

  @override
  Widget build(BuildContext context) {
    final enabled = onPressed != null;
    return Opacity(
      opacity: enabled ? 1 : .6,
      child: InkWell(
        onTap: onPressed,
        borderRadius: BorderRadius.circular(18),
        child: Ink(
          height: height,
          decoration: BoxDecoration(
            gradient: LinearGradient(colors: colors),
            borderRadius: BorderRadius.circular(18),
            boxShadow: [
              BoxShadow(
                color: colors.last.withOpacity(.35),
                blurRadius: 20,
                offset: const Offset(0, 8),
              )
            ],
          ),
          child: Center(
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(icon, color: Colors.white),
                const SizedBox(width: 10),
                Text(
                  label,
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.w700,
                    fontSize: 16,
                    letterSpacing: .2,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _GhostButton extends StatelessWidget {
  const _GhostButton({
    required this.onPressed,
    required this.icon,
    required this.label,
  });

  final VoidCallback onPressed;
  final IconData icon;
  final String label;

  @override
  Widget build(BuildContext context) {
    final border = BorderSide(color: Colors.black.withOpacity(.12));
    return OutlinedButton.icon(
      style: OutlinedButton.styleFrom(
        minimumSize: const Size.fromHeight(52),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
        side: border,
        foregroundColor: Colors.black87,
      ),
      onPressed: onPressed,
      icon: Icon(icon),
      label: Text(label, style: const TextStyle(fontWeight: FontWeight.w600)),
    );
  }
}
