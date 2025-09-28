// ─────────────────────────────────────────────────────────────────────────────
// lib/screens/exercise_preview_screen.dart
// Records video, lets you ANALYZE (offline on-device) and saves keypoints JSON.
// Adds Setup & Calibration management (front cam, tilt + 2D pose gating).
// ─────────────────────────────────────────────────────────────────────────────
import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:ui';

import 'package:amplify_auth_cognito/amplify_auth_cognito.dart';
import 'package:amplify_flutter/amplify_flutter.dart' as amp;
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart' show Listenable, kDebugMode;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:go_router/go_router.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:intl/intl.dart';
import 'package:path_provider/path_provider.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:share_plus/share_plus.dart';
import 'package:url_launcher/url_launcher.dart';

import '../shared/services/api_client.dart';
import '../shared/services/live_pose.dart';        // LIVE RTM-Pose 2D
import '../shared/services/pose_matcher.dart';
import '../shared/services/pose_runtime.dart';    // offline pipeline
import '../shared/services/s3_uploader.dart';
import '../shared/services/video_transcoder.dart';// 10fps helper
import '../shared/widgets/live_skeleton_overlay.dart';

class ExercisePreviewScreen extends StatefulWidget {
  const ExercisePreviewScreen({super.key, required this.exerciseId});
  final String exerciseId;

  @override
  State<ExercisePreviewScreen> createState() => _ExercisePreviewScreenState();
}

class _ExercisePreviewScreenState extends State<ExercisePreviewScreen>
    with TickerProviderStateMixin {
  late final CameraController _cam;
  bool _ready = false;
  bool _recording = false;
  bool _mirrorPreview = true; // front camera previews are mirrored visually
  final _picker = ImagePicker();

  // ─── Pose sanity checks toggles (0 = off, 1 = on). Turn on incrementally.
  static const int CHECK_FRAME_MARGIN        = 0;
  static const int CHECK_ASPECT_UPRIGHT      = 0;
  static const int CHECK_VERTICAL_ORDER      = 0;
  static const int CHECK_LEGS_UNDER_HIPS     = 0;
  static const int CHECK_SHOULDERS_OVER_HIPS = 0;
  static const int CHECK_ARMS_DOWN           = 0;
  static const int CHECK_SYMMETRY            = 0;

  // Reference-skeleton matcher (MAE in pixels)
  final PoseMatcher _matcher = PoseMatcher(
    fixedMaeThresholdPx: 30.0,
    dynamicMaePctOfRefH: 0.06,
  );

  bool _refReady = false;        // becomes true when reference JSON is loaded
  double? _lastMaePx;            // for debug/toast
  File? _lastVideo;              // last recorded or picked video
  XFile? _pendingRecording;      // temp holder for safely-stopped recordings

  /* ───────────── live 2D pose state ───────────── */
  final LivePoseEngine _engine = LivePoseEngine(yoloEvery: 5, roiMargin: 1.25);
  bool _streaming = false;
  bool _runningEst = false; // simple reentrancy guard
  bool _poseOk = false;     // true when detected 2D matches desired pose

  // Live 2D export buffer (ring buffer of recent frames)
  final List<Map<String, dynamic>> _liveKptLog = <Map<String, dynamic>>[];
  static const int _liveKptMax = 300; // ~ last few seconds depending on FPS

  /* ───────────── calibration (tilt + pose) ───────────── */
  // Live skeleton debug overlay (for on-screen QA)
  List<Offset>? _latestPts;
  Size? _latestImgSize;
  bool _showLiveSkeleton = true; // toggled via the AppBar eye icon
  List<Offset>? _refOverlayPts;  // reference skeleton shown in overlay (fixed)
  List<Offset>? _refFixedPts;    // same reference used for MAE comparisons
  late final AnimationController _skeletonColorCtrl;
  late final AnimationController _skeletonFadeCtrl;
  late final Animation<Color?> _skeletonColorAnim;
  late final Listenable _skeletonAnimation;
  bool _skeletonRefVisible = true;
  bool _skeletonHiddenAfterMatch = false;
  StreamSubscription<AccelerometerEvent>? _accelSub;

  // Tilt range caps
  static const double kTiltLow = 15.0;
  static const double kTiltHigh = 30.0;

  double _pitchDeg = 0; // +deg = phone top tilted upward

  bool get _tiltInRange => _pitchDeg >= kTiltLow && _pitchDeg <= kTiltHigh;
  bool get _tiltTooLow  => _pitchDeg < kTiltLow;
  bool get _tiltTooHigh => _pitchDeg > kTiltHigh;

  // Overlay controller (red glow → green, then fade out)
  late final AnimationController _glowCtrl;  // pulsing red
  late final AnimationController _flashCtrl; // flashing tilt icon
  bool _calibLockedGreen = false; // turn outline green
  bool _hideOutline = false;      // after success → disappear
  Timer? _holdTimer;              // 2-second hold once conditions satisfied

  @override
  void initState() {
    super.initState();
    _glowCtrl = AnimationController(
      vsync: this,
      lowerBound: 0.35,
      upperBound: 1.0,
      duration: const Duration(milliseconds: 900),
    )..repeat(reverse: true);
    _flashCtrl = AnimationController(
      vsync: this,
      lowerBound: .2,
      upperBound: 1,
      duration: const Duration(milliseconds: 700),
    )..repeat(reverse: true);

    _skeletonColorCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 350),
    );
    _skeletonFadeCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500),
    );
    _skeletonColorAnim = ColorTween(
      begin: Colors.redAccent,
      end: Colors.greenAccent,
    ).animate(CurvedAnimation(
      parent: _skeletonColorCtrl,
      curve: Curves.easeInOut,
    ));
    _skeletonAnimation = Listenable.merge([
      _skeletonColorCtrl,
      _skeletonFadeCtrl,
    ]);
    _skeletonFadeCtrl.addStatusListener((status) {
      if (!mounted) return;
      if (status == AnimationStatus.completed && _poseOk) {
        setState(() {
          _skeletonHiddenAfterMatch = true;
        });
      }
    });

    _initTiltStream();
    _boot(); // async bootstrap (awaits reference + camera, then starts stream)
  }

  /// Boot sequence: load reference JSON first, then init camera, then start stream.
  Future<void> _boot() async {
    try {
      await _matcher.ensureLoaded();
      _refReady = _matcher.refPoints != null;
      if (kDebugMode) {
        debugPrint('[ExercisePreview] reference loaded: $_refReady '
            '(kpts=${_matcher.refPoints?.length ?? 0}, '
            'source=${_matcher.refSource ?? 'unknown'})');
      }
    } catch (e) {
      if (kDebugMode) debugPrint('[ExercisePreview] ensureLoaded error: $e');
    }

    await _initCamera();

    // Start live 2D stream only after both camera and (ideally) reference are ready
    await _setStreaming(true);
  }

  // ───────────── sensors/tilt ─────────────
  void _initTiltStream() {
    const alpha = 0.85; // low-pass smoothing (higher = smoother/slower)
    _accelSub?.cancel();
    _accelSub = accelerometerEvents.listen((e) {
      final ay = e.y;
      final az = e.z;
      final rad = math.atan2(az, ay);
      final deg = (rad.abs() * 180.0 / math.pi).clamp(0.0, 90.0);

      // Smooth toward new value
      _pitchDeg = alpha * _pitchDeg + (1 - alpha) * deg;

      if (mounted) setState(() {}); // repaint label
      _updateCalibrationHold();     // keep gating responsive
    });
  }

  // Called whenever tilt or pose status changes
  void _updateCalibrationHold() {
    final ready = _tiltInRange && _poseOk; // BOTH must be true
    if (_hideOutline) return; // already done

    if (ready) {
      _holdTimer ??= Timer(const Duration(seconds: 2), () {
        if (!mounted) return;
        setState(() {
          _calibLockedGreen = true;
        });
        // After a small green confirmation, fade outline away.
        Future.delayed(const Duration(milliseconds: 800), () {
          if (!mounted) return;
          setState(() {
            _hideOutline = true;
          });
        });
      });
    } else {
      _holdTimer?.cancel();
      _holdTimer = null;
      if (_calibLockedGreen) return; // keep green until hidden
      if (_hideOutline) return;
      setState(() {}); // keep glowing red
    }
  }

  void _handleSkeletonVisualStateLocked(bool ok) {
    if (ok) {
      _skeletonHiddenAfterMatch = false;
      _skeletonRefVisible = false;
      _skeletonFadeCtrl.forward(from: 0.0);
      _skeletonColorCtrl.forward(from: 0.0);
    } else {
      _skeletonFadeCtrl.stop();
      _skeletonFadeCtrl.value = 0.0;
      _skeletonColorCtrl.stop();
      _skeletonColorCtrl.value = 0.0;
      _skeletonHiddenAfterMatch = false;
      _skeletonRefVisible = true;
    }
  }

  /* ───────────── live image stream → 2D RTM-Pose ───────────── */
  Future<void> _setStreaming(bool enable) async {
    if (!mounted || !_cam.value.isInitialized) return;
    if (enable && !_streaming) {
      await _cam.startImageStream((CameraImage img) {
        // ignore returned Future
        _onImageFromCamera(img);
      });
      _streaming = true;
    } else if (!enable && _streaming) {
      await _cam.stopImageStream();
      _streaming = false;
    }
  }

  Future<void> _onImageFromCamera(CameraImage img) async {
    if (_runningEst) return;
    _runningEst = true;
    try {
      final pts = await _engine.estimate2D(img); // Offsets in raw camera space

      // Log for export (if we have points)
      if (pts != null) {
        final frame = {
          't': DateTime.now().toIso8601String(),
          'pts': pts.map((p) => [p.dx, p.dy]).toList(growable: false),
        };
        _liveKptLog.add(frame);
        if (_liveKptLog.length > _liveKptMax) {
          _liveKptLog.removeAt(0);
        }
      }

      // Store for overlay drawing (live + reference projected to live)
      if (mounted) {
        if (pts != null) {
          _latestPts = pts;
          _latestImgSize = Size(img.width.toDouble(), img.height.toDouble());

          if (_matcher.refPoints != null) {
            // Always display (and compare against) a fixed, centered reference pose.
            _refOverlayPts =
                _matcher.centeredRefForFrame(_latestImgSize!, heightFrac: 0.70);
            _refFixedPts = _refOverlayPts;
            _refReady = _refOverlayPts != null;
          } else {
            _refOverlayPts = null;
            _refFixedPts = null;
            _refReady = false;
          }
        } else {
          _refOverlayPts = null; // avoid showing stale reference
          _refFixedPts = null;
          _refReady = false;
        }
        setState(() {}); // repaint overlay
      }

      // Pose gating via MAE-to-reference (primary) + optional sanity toggles
      final ok = _matchesDesiredPose(pts, img.width, img.height);

      if (!mounted) return;
      if (ok != _poseOk) {
        if (kDebugMode) {
          debugPrint('MAE(px) = ${_lastMaePx?.toStringAsFixed(1)}; ok=$ok');
        }
        setState(() {
          _poseOk = ok;
          _handleSkeletonVisualStateLocked(ok);
        });
        _updateCalibrationHold();
      }
    } catch (e) {
      // On any error, be conservative
      if (mounted && _poseOk) {
        setState(() {
          _poseOk = false;
          _handleSkeletonVisualStateLocked(false);
        });
        _updateCalibrationHold();
      }
      debugPrint('live 2D pose failed: $e');
    } finally {
      _runningEst = false;
    }
  }

  // Geometric test + MAE-to-reference matching.
  // - The MAE matcher is the primary gate (pixels).
  // - The existing sanity checks can be toggled via the CHECK_* flags.
  bool _matchesDesiredPose(List<Offset>? pts, int w, int h) {
    if (pts == null || pts.length < 17) {
      _lastMaePx = null;
      return false;
    }

    // 1) Primary: reference-skeleton MAE in pixels
    // Ensure we have a centered reference in the same camera space.
    _refFixedPts ??=
        _matcher.centeredRefForFrame(Size(w.toDouble(), h.toDouble()), heightFrac: 0.70);
    final mae = (_refFixedPts != null)
        ? _matcher.compareMAEPxAgainst(pts, _refFixedPts!)
        : double.infinity;
    _lastMaePx = mae;
    bool ok = _matcher.isMatchByMAE(mae);

    if (kDebugMode) {
      final refSrc = _matcher.refSource ?? 'unknown';
      final maeLabel = mae.isFinite ? mae.toStringAsFixed(2) : mae.toString();
      debugPrint('[PoseMatch] maePx=$maeLabel, match=$ok, '
          'reference=$refSrc, liveSpace=${w}x$h (raw camera)');
    }

    // 2) Optional sanity checks (each gated by a 0/1 flag from this screen)
    if (ok && (
        CHECK_FRAME_MARGIN == 1 || CHECK_ASPECT_UPRIGHT == 1 ||
        CHECK_VERTICAL_ORDER == 1 || CHECK_LEGS_UNDER_HIPS == 1 ||
        CHECK_SHOULDERS_OVER_HIPS == 1 || CHECK_ARMS_DOWN == 1 ||
        CHECK_SYMMETRY == 1)) {

      // Safe frame margin
      if (CHECK_FRAME_MARGIN == 1) {
        const m = 8.0;
        final req = [0,5,6,7,8,9,10,11,12,13,14,15,16];
        for (final i in req) {
          final p = pts[i];
          if (p.dx < m || p.dx > w - m || p.dy < m || p.dy > h - m) {
            ok = false; break;
          }
        }
      }

      if (ok && CHECK_ASPECT_UPRIGHT == 1) {
        final xs = pts.map((p) => p.dx).toList()..sort();
        final ys = pts.map((p) => p.dy).toList()..sort();
        final bboxW = xs.last - xs.first;
        final bboxH = ys.last - ys.first;
        if (bboxH < h * 0.45) ok = false;
        if (bboxH / (bboxW + 1e-3) < 1.20) ok = false;
      }

      if (ok && CHECK_VERTICAL_ORDER == 1) {
        final nose       = pts[0].dy;
        final shouldersY = (pts[5].dy + pts[6].dy) * 0.5;
        final hipsY      = (pts[11].dy + pts[12].dy) * 0.5;
        final kneesY     = (pts[13].dy + pts[14].dy) * 0.5;
        final anklesY    = (pts[15].dy + pts[16].dy) * 0.5;
        final vTol = 0.015 * h;
        bool ordered(a,b) => a < b + vTol;
        ok = ordered(nose, shouldersY) &&
             ordered(shouldersY, hipsY) &&
             ordered(hipsY, kneesY) &&
             ordered(kneesY, anklesY);
      }

      if (ok && CHECK_LEGS_UNDER_HIPS == 1) {
        final hipL = pts[11], hipR = pts[12];
        final ankL = pts[15], ankR = pts[16];
        final shoulderSpan = (pts[6].dx - pts[5].dx).abs();
        final legXTol = 0.60 * shoulderSpan;
        if ((ankL.dx - hipL.dx).abs() > legXTol) ok = false;
        if (ok && (ankR.dx - hipR.dx).abs() > legXTol) ok = false;
      }

      if (ok && CHECK_SHOULDERS_OVER_HIPS == 1) {
        final shC = (pts[5].dx + pts[6].dx) * 0.5;
        final hipC= (pts[11].dx + pts[12].dx) * 0.5;
        final shoulderSpan = (pts[6].dx - pts[5].dx).abs();
        if ((shC - hipC).abs() > 0.35 * shoulderSpan) ok = false;
      }

      if (ok && CHECK_ARMS_DOWN == 1) {
        final elbL = pts[7], elbR = pts[8];
        final wriL = pts[9], wriR = pts[10];
        final hipL = pts[11], hipR = pts[12];
        final shoulderSpan = (pts[6].dx - pts[5].dx).abs();
        final wristNearXTol = 0.80 * shoulderSpan;
        final bboxH = (pts.map((p) => p.dy).reduce(math.max) -
                       pts.map((p) => p.dy).reduce(math.min));
        final wristNearYTol = 0.35 * bboxH;
        final hipsY = (hipL.dy + hipR.dy) * 0.5;

        final wristsBelowElbows =
          (wriL.dy > elbL.dy + 0.02 * h) && (wriR.dy > elbR.dy + 0.02 * h);
        final wristsNearTorsoX =
          (wriL.dx - hipL.dx).abs() < wristNearXTol &&
          (wriR.dx - hipR.dx).abs() < wristNearXTol;
        final wristsNearHipsY =
          (wriL.dy - hipsY).abs() < wristNearYTol &&
          (wriR.dy - hipsY).abs() < wristNearYTol;

        ok = wristsBelowElbows && wristsNearTorsoX && wristsNearHipsY;
      }

      if (ok && CHECK_SYMMETRY == 1) {
        final yTol = 0.12 * (pts.map((p) => p.dy).reduce(math.max) -
                             pts.map((p) => p.dy).reduce(math.min));
        final sym =
          (pts[5].dy - pts[6].dy).abs() < yTol &&
          (pts[11].dy - pts[12].dy).abs() < yTol &&
          (pts[15].dy - pts[16].dy).abs() < yTol;
        ok = sym;
      }
    }

    return ok;
  }

  /* ───────────── progress helper ───────────── */
  Future<void> _showProcessingDialog() async {
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
                child: CircularProgressIndicator(strokeWidth: 3)),
            SizedBox(width: 24),
            Expanded(child: Text('Analyzing your video…')),
          ],
        ),
      ),
    );
  }

  void _hideProcessingDialog() {
    if (!mounted) return;
    Navigator.of(context, rootNavigator: false)
        .popUntil((route) => route is! PopupRoute);
  }

  Future<File?> pickVideoFromGallery() async {
    final xFile = await _picker.pickVideo(source: ImageSource.gallery);
    return xFile == null ? null : File(xFile.path);
  }

  Future<void> _rememberVideo(File videoFile) async {
    try {
      final dir = await getApplicationDocumentsDirectory();
      final ts = DateFormat('yyyy-MM-dd_HH-mm-ss').format(DateTime.now());
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
    await file
        .writeAsString(const JsonEncoder.withIndent('  ').convert(report));
    return file.path;
  }

  Future<void> _analyzeOffline() async {
    final file = _lastVideo;
    if (file == null) return;

    try {
      _showProcessingDialog();
      final pipeline = PosePipeline();
      final res = await pipeline.analyzeVideo(file);

      final report = res.toReport(); // now contains kpts
      final savedPath = await _saveLocalReport(report);
      _hideProcessingDialog();

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
              content:
                  Text('✅ Saved keypoints → ${savedPath.split('/').last}')),
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
    final ts = DateFormat('yyyy-MM-dd_HH-mm-ss').format(DateTime.now());
    final s3Path = 'private/videos/${widget.exerciseId}/$ts.mp4';

    // A. local copy (for playback)
    String localPath = videoFile.path;
    try {
      final dir = await getApplicationDocumentsDirectory();
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
          action: SnackBarAction(
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
        'videoKey': s3Path,
        'report': report,
      },
    );
  }

  Future<void> _exportLive2D() async {
    if (_liveKptLog.isEmpty) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('No live 2D data to export yet.')),
      );
      return;
    }
    final dir = await getApplicationDocumentsDirectory();
    final folder = Directory('${dir.path}/live2d/${widget.exerciseId}');
    await folder.create(recursive: true);
    final ts = DateFormat('yyyy-MM-dd_HH-mm-ss').format(DateTime.now());
    final file = File('${folder.path}/$ts.json');

    final payload = {
      'exerciseId': widget.exerciseId,
      'source': 'preview_live_rtm_pose_2d',
      'frames': _liveKptLog,
    };
    await file.writeAsString(const JsonEncoder.withIndent('  ').convert(payload));

    if (!mounted) return;

    // Toast + iOS share sheet (AirDrop, Mail, etc.)
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('✅ Exported live 2D → ${file.path.split('/').last}')),
    );

    try {
      await Share.shareXFiles(
        [XFile(file.path)],
        text: 'Live 2D RTM-Pose export (${widget.exerciseId})',
        subject: 'Live 2D RTM-Pose export',
      );
    } catch (e) {
      debugPrint('Share error: $e');
    }
  }

  /* ─────────────────── CAMERA ─────────────────── */
  Future<void> _stopRecordingSafely() async {
    CameraController? controller;
    try {
      controller = _cam;
    } catch (_) {
      controller = null;
    }

    final c = controller;
    if (c == null) return;

    if (c.value.isStreamingImages) {
      try {
        await c.stopImageStream();
        _streaming = false;
      } catch (_) {}
      await Future.delayed(const Duration(milliseconds: 30));
    }

    if (c.value.isRecordingVideo) {
      try {
        final xFile = await c.stopVideoRecording();
        _pendingRecording = xFile;
      } catch (_) {}
    }
  }

  Future<void> _initCamera() async {
    final cams = await availableCameras();
    final selected = cams.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.front,
      orElse: () => cams.first,
    );

    _cam = CameraController(
      selected, // front preferred for preview/recording
      ResolutionPreset.high,
      enableAudio: true,
    );

    await _cam.initialize();
    await _cam.lockCaptureOrientation(DeviceOrientation.portraitUp);

    // Mirror overlay to match preview for front cam; not mirrored for back cam.
    _mirrorPreview = selected.lensDirection == CameraLensDirection.front;

    if (mounted) setState(() => _ready = true);

    // NOTE: We no longer start streaming here; _boot() does it AFTER reference loads.
  }

  @override
  void dispose() {
    _holdTimer?.cancel();
    _accelSub?.cancel();
    _flashCtrl.dispose();
    _glowCtrl.dispose();
    _skeletonColorCtrl.dispose();
    _skeletonFadeCtrl.dispose();
    _stopRecordingSafely();
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
        actions: [
          IconButton(
            icon: const Icon(Icons.download_rounded),
            tooltip: 'Export 2D (preview)',
            onPressed: _exportLive2D,
          ),
          IconButton(
            tooltip: _showLiveSkeleton ? 'Hide skeleton' : 'Show skeleton',
            icon: Icon(_showLiveSkeleton ? Icons.visibility : Icons.visibility_off),
            onPressed: () => setState(() => _showLiveSkeleton = !_showLiveSkeleton),
          ),
        ],
      ),
      body: Column(
        children: [
          // Camera preview + calibration overlays
          Expanded(
            child: _ready
                ? FittedBox(
                    fit: BoxFit.cover,
                    child: SizedBox(
                      width: _cam.value.previewSize!.height,
                      height: _cam.value.previewSize!.width,
                      child: Stack(
                        fit: StackFit.expand,
                        children: [
                          CameraPreview(_cam),

                          // Live skeleton overlay (debug)
                          if (_showLiveSkeleton &&
                              _latestPts != null &&
                              _latestImgSize != null &&
                              !_skeletonHiddenAfterMatch)
                            AnimatedBuilder(
                              animation: _skeletonAnimation,
                              builder: (context, _) {
                                final opacity = (1.0 - _skeletonFadeCtrl.value)
                                    .clamp(0.0, 1.0);
                                if (opacity <= 0.0) {
                                  return const SizedBox.shrink();
                                }
                                final color =
                                    _skeletonColorAnim.value ?? Colors.redAccent;
                                return Opacity(
                                  opacity: opacity,
                                  child: LiveSkeletonOverlay(
                                    points: _latestPts,
                                    imageSize: _latestImgSize!,
                                    referencePoints: _skeletonRefVisible
                                        ? _refOverlayPts
                                        : null,
                                    mirrorHorizontally: false,
                                    color: color,
                                    thickness: 3.5,
                                    showJoints: true,
                                    boxFit: BoxFit.cover,
                                    alignment: Alignment.center,
                                  ),
                                );
                              },
                            ),

                          // Tilt guidance (bigger icon/text; 15–30° target; simplified messages)
                          Align(
                            alignment: Alignment.topCenter,
                            child: Padding(
                              padding: const EdgeInsets.only(top: 16),
                              child: AnimatedBuilder(
                                animation: _flashCtrl,
                                builder: (_, __) {
                                  final bool outOfRange = _tiltTooLow || _tiltTooHigh;
                                  final double flashingOpacity =
                                      outOfRange ? (0.55 + 0.45 * _flashCtrl.value) : 1.0;

                                  final IconData icon = _tiltTooLow
                                      ? Icons.north_rounded
                                      : (_tiltTooHigh ? Icons.south_rounded : Icons.check_circle_rounded);

                                  final Color iconColor = _tiltTooLow || _tiltTooHigh
                                      ? Colors.orangeAccent
                                      : Colors.greenAccent;

                                  final String label = _tiltTooLow
                                      ? 'Increase tilt'
                                      : (_tiltTooHigh ? 'Decrease tilt' : 'Perfect');

                                  return Opacity(
                                    opacity: flashingOpacity,
                                    child: Column(
                                      mainAxisSize: MainAxisSize.min,
                                      children: [
                                        Icon(icon, size: 56, color: iconColor), // bigger icon
                                        const SizedBox(height: 10),
                                        Container(
                                          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                                          decoration: BoxDecoration(
                                            color: Colors.black.withOpacity(.38),
                                            borderRadius: BorderRadius.circular(14),
                                          ),
                                          child: Text(
                                            label,
                                            textAlign: TextAlign.center,
                                            style: const TextStyle(
                                              color: Colors.white,
                                              fontWeight: FontWeight.w800,
                                              fontSize: 20, // bigger font
                                              letterSpacing: .2,
                                            ),
                                          ),
                                        ),
                                      ],
                                    ),
                                  );
                                },
                              ),
                            ),
                          ),

                          // Calibration outline overlay (glowing red → green → fade)
                          if (!_hideOutline)
                            IgnorePointer(
                              child: AnimatedOpacity(
                                duration: const Duration(milliseconds: 350),
                                opacity: _calibLockedGreen ? 0.0 : 1.0,
                                child: CustomPaint(
                                  painter: _NPoseOutlinePainter(
                                    pulse: _glowCtrl.value,
                                    color: _calibLockedGreen
                                        ? Colors.greenAccent
                                        : Colors.redAccent,
                                    isGreen: _calibLockedGreen,
                                  ),
                                ),
                              ),
                            ),

                          // If we locked green, draw a final non-pulsing green outline for ~800ms.
                          if (_calibLockedGreen && !_hideOutline)
                            CustomPaint(
                              painter: _NPoseOutlinePainter(
                                pulse: 1.0,
                                color: Colors.greenAccent,
                                isGreen: true,
                              ),
                            ),

                          // Bottom gradient to improve legibility of actions over video
                          const _BottomScrim(),
                        ],
                      ),
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
      // Stop recording → resume live stream
      _pendingRecording = null;
      await _stopRecordingSafely();
      setState(() => _recording = false);
      final xFile = _pendingRecording;
      _pendingRecording = null;
      if (xFile != null) {
        await _rememberVideo(File(xFile.path));
      }
      if (!_streaming) await _setStreaming(true);
      HapticFeedback.selectionClick();
    } else {
      // Start recording → stop live stream (Camera cannot do both)
      if (_streaming) await _setStreaming(false);
      await _cam.prepareForVideoRecording();
      _pendingRecording = null;
      await _cam.startVideoRecording();
      setState(() => _recording = true);
      HapticFeedback.heavyImpact();
    }
  }

  // POST { "s3_key": "<path>", … } to Flask /enqueue
  Future<void> _notifyBackend(String s3Path) async {
    const backendEndpoint = 'http://63.178.80.242:5001/enqueue';
    try {
      final sess = await amp.Amplify.Auth
          .fetchAuthSession() as CognitoAuthSession;
      final jwt = sess.userPoolTokensResult.valueOrNull?.accessToken.raw;

      final resp = await http.post(
        Uri.parse(backendEndpoint),
        headers: {
          'Content-Type': 'application/json',
          if (jwt != null) 'Authorization': 'Bearer $jwt',
        },
        body: jsonEncode({
          's3_key': s3Path,
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
        'squat' => Icons.fitness_center,
        'deadlift' => Icons.accessibility_new,
        _ => Icons.sports_gymnastics,
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

/* ───────── Visual helpers ───────── */

class _BottomScrim extends StatelessWidget {
  const _BottomScrim();

  @override
  Widget build(BuildContext context) {
    return IgnorePointer(
      child: Align(
        alignment: Alignment.bottomCenter,
        child: Container(
          height: 120,
          decoration: const BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment(0, .1),
              end: Alignment.bottomCenter,
              colors: [Colors.transparent, Color.fromARGB(140, 0, 0, 0)],
            ),
          ),
        ),
      ),
    );
  }
}

/// Painter for a continuous "N-pose" silhouette overlay (no live skeleton).
/// Pulses red while calibrating; turns green then fades when locked.
class _NPoseOutlinePainter extends CustomPainter {
  _NPoseOutlinePainter({
    required this.pulse,
    required this.color,
    required this.isGreen,
  });

  final double pulse; // 0..1
  final Color color;
  final bool isGreen;

  @override
  void paint(Canvas canvas, Size size) {
    final w = size.width;
    final h = size.height;
    final cx = w / 2;

    // Layout ratios for a standing person
    final headR      = h * 0.060;
    final headCY     = h * 0.22;
    final shoulderY  = headCY + headR + h * 0.020;
    final shoulderHW = w * 0.20;      // half shoulder width
    final waistY     = shoulderY + h * 0.14;
    final hipY       = waistY + h * 0.10;
    final footY      = h * 0.86;

    final waistHW    = shoulderHW * 0.70;
    final hipHW      = shoulderHW * 0.88;

    final armT       = w * 0.055;
    final handY      = hipY - h * 0.08;

    final legGap     = w * 0.085;
    final legT       = w * 0.095;

    // ---- Build filled shapes for each body part ----
    final Path head = Path()
      ..addOval(Rect.fromCircle(center: Offset(cx, headCY), radius: headR));

    // Torso as a single closed, curved shape (shoulders → waist pinch → hips)
    final Path torso = Path()
      ..moveTo(cx + shoulderHW, shoulderY)
      ..cubicTo(
        cx + shoulderHW, shoulderY + h * 0.06,
        cx + waistHW,    waistY - h * 0.02,
        cx + waistHW,    waistY,
      )
      ..cubicTo(
        cx + waistHW, waistY + h * 0.02,
        cx + hipHW,   hipY - h * 0.01,
        cx + hipHW,   hipY,
      )
      ..lineTo(cx - hipHW, hipY)
      ..cubicTo(
        cx - hipHW, hipY - h * 0.01,
        cx - waistHW, waistY + h * 0.02,
        cx - waistHW, waistY,
      )
      ..cubicTo(
        cx - waistHW, waistY - h * 0.02,
        cx - shoulderHW, shoulderY + h * 0.06,
        cx - shoulderHW, shoulderY,
      )
      ..close();

    // Arms (slim rounded rectangles)
    final Path leftArm = Path()
      ..addRRect(RRect.fromRectAndRadius(
        Rect.fromLTRB(
          cx - shoulderHW - armT, shoulderY,
          cx - shoulderHW,        handY,
        ),
        Radius.circular(armT * 0.6),
      ));
    final Path rightArm = Path()
      ..addRRect(RRect.fromRectAndRadius(
        Rect.fromLTRB(
          cx + shoulderHW,        shoulderY,
          cx + shoulderHW + armT, handY,
        ),
        Radius.circular(armT * 0.6),
      ));

    // Legs (rounded rectangles) with a center gap
    final Path leftLeg = Path()
      ..addRRect(RRect.fromRectAndRadius(
        Rect.fromLTRB(
          cx - legGap - legT, hipY,
          cx - legGap,        footY,
        ),
        Radius.circular(legT * 0.45),
      ));
    final Path rightLeg = Path()
      ..addRRect(RRect.fromRectAndRadius(
        Rect.fromLTRB(
          cx + legGap,        hipY,
          cx + legGap + legT, footY,
        ),
        Radius.circular(legT * 0.45),
      ));

    // Union all shapes to get a single continuous silhouette region
    Path silhouette = Path.combine(PathOperation.union, head, torso);
    silhouette = Path.combine(PathOperation.union, silhouette, leftArm);
    silhouette = Path.combine(PathOperation.union, silhouette, rightArm);
    silhouette = Path.combine(PathOperation.union, silhouette, leftLeg);
    silhouette = Path.combine(PathOperation.union, silhouette, rightLeg);

    // ---- Paints: glow + crisp outline ----
    final double glowAlpha = isGreen ? 0.90 : (0.70 + 0.30 * pulse);
    final double glowWidth = (isGreen ? 9.0 : 8.0 + 6.0 * pulse);

    final Paint glow = Paint()
      ..color = color.withOpacity(glowAlpha)
      ..style = PaintingStyle.stroke
      ..strokeWidth = glowWidth
      ..strokeJoin = StrokeJoin.round
      ..maskFilter = const MaskFilter.blur(BlurStyle.outer, 12);

    final Paint line = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = isGreen ? 4.5 : 3.6
      ..strokeJoin = StrokeJoin.round;

    // Draw glow first, then crisp outline
    canvas.drawPath(silhouette, glow);
    canvas.drawPath(silhouette, line);
  }

  @override
  bool shouldRepaint(covariant _NPoseOutlinePainter oldDelegate) {
    return oldDelegate.pulse != pulse ||
        oldDelegate.color != color ||
        oldDelegate.isGreen != isGreen;
  }
}
