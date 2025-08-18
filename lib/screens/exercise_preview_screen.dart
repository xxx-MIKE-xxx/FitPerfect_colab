// lib/screens/exercise_preview_screen.dart
// ─────────────────────────────────────────────────────────────────────────────
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

  // remember the last captured/selected video so the user can choose what to do
  File? _lastVideo;

  /* ───────────── progress helper ───────────── */
  Future<void> _showProcessingDialog() async {
    debugPrint('[ExercisePreview] showing “Analyzing…” dialog');
    await showDialog(
      context: context,
      barrierDismissible: false,
      useRootNavigator: false,
      builder: (_) => const AlertDialog(
        content: Row(
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

  // Save a stable local copy and remember it for later actions.
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
      debugPrint('[ExercisePreview] local saved copy → ${saved.path}');
    } catch (e) {
      // Fall back to temp file if copy fails.
      setState(() => _lastVideo = videoFile);
      debugPrint('[ExercisePreview] could not save local copy: $e');
    }
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
      _showProcessingDialog();                       // fire-and-forget

      final report = await ApiClient.fetchReport(
        s3Path,
        delay: const Duration(seconds: 6),
        max: 120,
      );

      _hideProcessingDialog();
      debugPrint('✅ report downloaded for $s3Path');

      _navigateToFeedback(localPath, s3Path, report);
      if (mounted) {
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

  // bypass helper (no upload, no auth)
  void _bypassToFeedback() {
    final file = _lastVideo;
    if (file == null) return;
    final placeholderKey =
        'local/${widget.exerciseId}/${DateTime.now().millisecondsSinceEpoch}.mp4';
    _navigateToFeedback(file.path, placeholderKey, const {'data': []});
  }

  // user-triggered analysis
  Future<void> _analyzeNow() async {
    final file = _lastVideo;
    if (file == null) return;
    if (!await _ensureSignedIn()) return;
    await _uploadAndProcess(file);
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
      backgroundColor: const Color(0xFFF7F9FB),
      appBar: AppBar(
        elevation: 0,
        backgroundColor: const Color(0xFFF7F9FB),
        title: Row(
          children: [
            Container(
              width: 36,
              height: 36,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                gradient: const LinearGradient(
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
                ? Stack(
                    children: [
                      // Keep the same orientation-preserving preview
                      FittedBox(
                        fit: BoxFit.cover,
                        child: SizedBox(
                          width : _cam.value.previewSize!.height,
                          height: _cam.value.previewSize!.width,
                          child : CameraPreview(_cam),
                        ),
                      ),
                      // Status chip
                      Positioned(
                        top: 12,
                        right: 12,
                        child: _StatusChip(
                          recording: _recording,
                          videoReady: _lastVideo != null,
                        ),
                      ),
                    ],
                  )
                : const Center(child: CircularProgressIndicator()),
          ),

          // Action panel (visual revamp, same actions)
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
                  await _rememberVideo(file);   // ← just remember; no upload
                },
                onAnalyze: _analyzeNow,
                onSkip: _bypassToFeedback,
              ),
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
    if (_recording) {
      final xFile = await _cam.stopVideoRecording();
      setState(() => _recording = false);
      await _rememberVideo(File(xFile.path)); // ← just remember; no upload
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

/* ───────── UI widgets (visual only) ───────── */

class _StatusChip extends StatelessWidget {
  const _StatusChip({required this.recording, required this.videoReady});
  final bool recording;
  final bool videoReady;

  @override
  Widget build(BuildContext context) {
    final color = recording
        ? Colors.red
        : (videoReady ? const Color(0xFF00C853) : Colors.black54);
    final label = recording
        ? 'Recording…'
        : (videoReady ? 'Video ready' : 'Ready');

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(.92),
        borderRadius: BorderRadius.circular(24),
        boxShadow: const [
          BoxShadow(color: Color(0x1A000000), blurRadius: 12, offset: Offset(0, 4)),
        ],
      ),
      child: Row(
        children: [
          Container(
            width: 8, height: 8,
            decoration: BoxDecoration(color: color, shape: BoxShape.circle),
          ),
          const SizedBox(width: 8),
          Text(label, style: TextStyle(fontWeight: FontWeight.w600, color: Colors.black87)),
        ],
      ),
    );
  }
}

class _ActionPanel extends StatelessWidget {
  const _ActionPanel({
    required this.recording,
    required this.hasVideo,
    required this.onToggleRecord,
    required this.onPickVideo,
    required this.onAnalyze,
    required this.onSkip,
  });

  final bool recording;
  final bool hasVideo;
  final VoidCallback? onToggleRecord;
  final VoidCallback onPickVideo;
  final VoidCallback onAnalyze;
  final VoidCallback onSkip;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

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
            // Record / Stop
            _GradientButton(
              height: 56,
              onPressed: onToggleRecord,
              colors: recording
                  ? const [Color(0xFFD32F2F), Color(0xFFE53935)]               // intense red while recording
                  : const [Color(0xFFFFC107), Color(0xFFFF7043)],              // amber → fire red
              icon: recording ? Icons.stop : Icons.fiber_manual_record,
              label: recording ? 'Stop recording' : 'Tap to record',
            ),
            const SizedBox(height: 10),
            // Pick from gallery
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
                      colors: const [Color(0xFF00BFA5), Color(0xFF1DE9B6)],   // teal → mint
                      icon: Icons.analytics,
                      label: 'Analyze now',
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: _GhostButton(
                      onPressed: onSkip,
                      icon: Icons.skip_next,
                      label: 'Continue without analysis',
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
      label: Text(
        label,
        style: const TextStyle(fontWeight: FontWeight.w600),
      ),
    );
  }
}
