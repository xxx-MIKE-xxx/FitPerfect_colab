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
import 'package:flutter/painting.dart' show applyBoxFit;
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
import '../shared/services/live_pose.dart'; // LIVE RTM-Pose 2D
import '../shared/services/pose_matcher.dart';
import '../shared/services/pose_processing_controller.dart';
import '../shared/services/pose_runtime.dart'; // offline pipeline
import '../shared/services/pose_session_storage.dart';
import '../shared/services/s3_uploader.dart';
import '../shared/services/video_transcoder.dart'; // 10fps helper
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
  static const int CHECK_FRAME_MARGIN = 0;
  static const int CHECK_ASPECT_UPRIGHT = 0;
  static const int CHECK_VERTICAL_ORDER = 0;
  static const int CHECK_LEGS_UNDER_HIPS = 0;
  static const int CHECK_SHOULDERS_OVER_HIPS = 0;
  static const int CHECK_ARMS_DOWN = 0;
  static const int CHECK_SYMMETRY = 0;

  // Reference-skeleton matcher (MAE in pixels)
  final PoseMatcher _matcher = PoseMatcher(
    fixedMaeThresholdPx: 30.0,
    dynamicMaePctOfRefH: 0.06,
  );

  bool _refReady = false; // becomes true when reference JSON is loaded
  double? _lastMaePx; // for debug/toast
  File? _lastVideo; // last recorded or picked video
  XFile? _pendingRecording; // temp holder for safely-stopped recordings
  DateTime? _recordingStartedAt;

  late final PoseProcessingController _poseProcessingController;
  StreamSubscription<ProgressEvent>? _poseProgressSub;
  ProgressEvent _processingEvent = ProgressEvent.hidden();
  bool _processingActive = false;
  String? _activeSessionId;

  /* ───────────── live 2D pose state ───────────── */
  final LivePoseEngine _engine = LivePoseEngine(yoloEvery: 5, roiMargin: 1.25);
  bool _streaming = false;
  bool _runningEst = false; // simple reentrancy guard
  bool _poseOk = false; // true when detected 2D matches desired pose

  // Live 2D export buffer (ring buffer of recent frames)
  final List<Map<String, dynamic>> _liveKptLog = <Map<String, dynamic>>[];
  static const int _liveKptMax = 300; // ~ last few seconds depending on FPS

  /* ───────────── calibration (tilt + pose) ───────────── */
  // Live skeleton debug overlay (for on-screen QA)
  List<Offset>? _latestPts;
  Size? _latestImgSize;
  bool _showLiveSkeleton = true; // toggled via the AppBar eye icon
  List<Offset>? _refOverlayPts; // reference skeleton shown in overlay (fixed)
  List<Offset>? _refFixedPts; // same reference used for MAE comparisons
  late final AnimationController _skeletonColorCtrl;
  late final AnimationController _skeletonFadeCtrl;
  late final Animation<Color?> _skeletonColorAnim;
  late final Listenable _skeletonAnimation;
  bool _skeletonHiddenAfterMatch = false;
  bool _logPoseEvents = false;
  StreamSubscription<AccelerometerEvent>? _accelSub;

  // Tilt range caps
  static const double kTiltLow = 15.0;
  static const double kTiltHigh = 30.0;

  double _pitchDeg = 0; // +deg = phone top tilted upward

  bool get _tiltInRange => _pitchDeg >= kTiltLow && _pitchDeg <= kTiltHigh;
  bool get _tiltTooLow => _pitchDeg < kTiltLow;
  bool get _tiltTooHigh => _pitchDeg > kTiltHigh;

  // Overlay controller (red glow → green, then fade out)
  late final AnimationController _glowCtrl; // pulsing red
  late final AnimationController _flashCtrl; // flashing tilt icon
  bool _calibLockedGreen = false; // turn outline green
  bool _hideOutline = false; // after success → disappear
  Timer? _holdTimer; // 2-second hold once conditions satisfied

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

    _poseProcessingController = PoseProcessingController();
    _poseProgressSub =
        _poseProcessingController.progress.stream.listen((event) {
      if (!mounted) return;
      setState(() {
        _processingEvent = event;
        if (event.state == ProgressState.complete ||
            event.state == ProgressState.hidden ||
            event.state == ProgressState.error) {
          _processingActive = false;
        } else {
          _processingActive = true;
        }
      });
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
      _updateCalibrationHold(); // keep gating responsive
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
      _skeletonFadeCtrl.forward(from: 0.0);
      _skeletonColorCtrl.forward(from: 0.0);
    } else {
      _skeletonFadeCtrl.stop();
      _skeletonFadeCtrl.value = 0.0;
      _skeletonColorCtrl.stop();
      _skeletonColorCtrl.value = 0.0;
      _skeletonHiddenAfterMatch = false;
    }
  }

  /* ───────────── live image stream → 2D RTM-Pose ───────────── */
  Future<void> _setStreaming(bool enable) async {
    if (!mounted) return;
    if (!_cam.value.isInitialized) return;

    if (enable) {
      // Never stream while recording
      if (_cam.value.isRecordingVideo) return;
      if (!_cam.value.isStreamingImages) {
        await _cam.startImageStream((CameraImage img) {
          _onImageFromCamera(img); // fire-and-forget
        });
        _streaming = true;
      }
    } else {
      if (_cam.value.isStreamingImages) {
        try {
          await _cam.stopImageStream();
        } catch (_) {}
        _streaming = false;
        await Future.delayed(const Duration(milliseconds: 30)); // let native drain
      }
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
        if (kDebugMode && _logPoseEvents) {
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
      if (kDebugMode && _logPoseEvents) {
        debugPrint('live 2D pose failed: $e');
      }
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

    if (kDebugMode && _logPoseEvents) {
      final refSrc = _matcher.refSource ?? 'unknown';
      final maeLabel = mae.isFinite ? mae.toStringAsFixed(2) : mae.toString();
      debugPrint('[PoseMatch] maePx=$maeLabel, match=$ok, '
          'reference=$refSrc, liveSpace=${w}x$h (raw camera)');
    }

    // 2) Optional sanity checks (each gated by a 0/1 flag from this screen)
    if (ok &&
        (CHECK_FRAME_MARGIN == 1 ||
            CHECK_ASPECT_UPRIGHT == 1 ||
            CHECK_VERTICAL_ORDER == 1 ||
            CHECK_LEGS_UNDER_HIPS == 1 ||
            CHECK_SHOULDERS_OVER_HIPS == 1 ||
            CHECK_ARMS_DOWN == 1 ||
            CHECK_SYMMETRY == 1)) {
      // Safe frame margin
      if (CHECK_FRAME_MARGIN == 1) {
        const m = 8.0;
        final req = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        for (final i in req) {
          final p = pts[i];
          if (p.dx < m || p.dx > w - m || p.dy < m || p.dy > h - m) {
            ok = false;
            break;
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
        final nose = pts[0].dy;
        final shouldersY = (pts[5].dy + pts[6].dy) * 0.5;
        final hipsY = (pts[11].dy + pts[12].dy) * 0.5;
        final kneesY = (pts[13].dy + pts[14].dy) * 0.5;
        final anklesY = (pts[15].dy + pts[16].dy) * 0.5;
        final vTol = 0.015 * h;
        bool ordered(a, b) => a < b + vTol;
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
        final hipC = (pts[11].dx + pts[12].dx) * 0.5;
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
        final yTol = 0.12 *
            (pts.map((p) => p.dy).reduce(math.max) -
                pts.map((p) => p.dy).reduce(math.min));
        final sym = (pts[5].dy - pts[6].dy).abs() < yTol &&
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

  Future<File> _rememberVideo(File videoFile) async {
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
      return saved;
    } catch (_) {
      setState(() => _lastVideo = videoFile); // fall back
      return videoFile;
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

  Future<void> _processRecordedSession(
    File videoFile,
    DateTime startedAt,
    DateTime endedAt,
  ) async {
    final sessionId = PoseSessionWriter.newSessionId();
    setState(() {
      _processingActive = true;
      _processingEvent = ProgressEvent.indeterminate(phase: 'Preparing session…');
    });

    final request = PoseProcessingRequest(
      videoFile: videoFile,
      sessionId: sessionId,
      exerciseId: widget.exerciseId,
      startTime: startedAt,
      endTime: endedAt,
      exerciseLabel: widget.exerciseId.capitalize(),
      appVersion: null,
    );

    try {
      final outcome = await _poseProcessingController.process(request);
      if (!mounted) return;
      _activeSessionId = sessionId;
      await _navigateToFeedback(
        videoFile.path,
        'local/${widget.exerciseId}/$sessionId.mp4',
        const {'data': []},
        sessionId: sessionId,
        summary2d: outcome.summary2D,
        summary3d: outcome.summary3D,
        meta: outcome.meta,
      );
    } on PoseProcessingCancelled {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Processing cancelled.')),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('3D processing failed: $e')),
      );
    } finally {
      if (!mounted) return;
      setState(() {
        _processingActive = false;
        if (_processingEvent.state != ProgressState.error) {
          _processingEvent = ProgressEvent.hidden();
        }
      });
    }
  }

  Future<void> _promptCancelProcessing() async {
    if (!_processingActive) return;
    final confirm = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Cancel processing?'),
        content: const Text(
          'Cancelling will discard all pose data recorded for this session.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: const Text('Keep processing'),
          ),
          TextButton(
            onPressed: () => Navigator.of(context).pop(true),
            style: TextButton.styleFrom(foregroundColor: Colors.redAccent),
            child: const Text('Cancel session'),
          ),
        ],
      ),
    );
    if (confirm == true) {
      await _poseProcessingController.cancel();
      if (!mounted) return;
      setState(() {
        _processingActive = false;
        _processingEvent = ProgressEvent.hidden();
      });
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Processing cancelled. Session deleted.')),
      );
    }
  }

  void _handleProcessingCancel() {
    unawaited(_promptCancelProcessing());
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
      await _navigateToFeedback(
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
      final name = s3Path.replaceAll('/', '_');
      final saved = await videoFile.copy('${dir.path}/$name');
      localPath = saved.path;
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

      await _navigateToFeedback(localPath, s3Path, report);
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
  Future<void> _suspendCameraForRouteChange() async {
    if (!_cam.value.isInitialized) return;
    try {
      if (_cam.value.isStreamingImages) {
        await _cam.stopImageStream();
        _streaming = false;
        await Future.delayed(const Duration(milliseconds: 50));
      }
    } catch (_) {}
    try {
      if (_cam.value.isRecordingVideo) {
        final x = await _cam.stopVideoRecording();
        _pendingRecording = x;
      }
    } catch (_) {}
  }

  Future<void> _navigateToFeedback(
    String videoPath,
    String s3Path,
    Map<String, dynamic> report, {
    String? sessionId,
    Map<String, dynamic>? summary2d,
    Map<String, dynamic>? summary3d,
    Map<String, dynamic>? meta,
  }
  ) async {
    if (!mounted) return;
    await _suspendCameraForRouteChange();
    if (!mounted) return;
    context.go(
      '/feedback',
      extra: {
        'videoPath': videoPath,
        'videoKey': s3Path,
        'report': report,
        if (sessionId != null) 'sessionId': sessionId,
        if (summary2d != null) 'summary2D': summary2d,
        if (summary3d != null) 'summary3D': summary3d,
        if (meta != null) 'sessionMeta': meta,
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
    await file
        .writeAsString(const JsonEncoder.withIndent('  ').convert(payload));

    if (!mounted) return;

    // Toast + iOS share sheet (AirDrop, Mail, etc.)
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
          content: Text('✅ Exported live 2D → ${file.path.split('/').last}')),
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
      } catch (_) {}
      await Future.delayed(const Duration(milliseconds: 30));
      _streaming = false;
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
      imageFormatGroup:
          Platform.isIOS ? ImageFormatGroup.bgra8888 : ImageFormatGroup.yuv420,
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
    _poseProgressSub?.cancel();
    _flashCtrl.dispose();
    _glowCtrl.dispose();
    _skeletonColorCtrl.dispose();
    _skeletonFadeCtrl.dispose();
    _stopRecordingSafely();
    _logPoseEvents = false;
    _cam.dispose();
    super.dispose();
  }

  /* ───────────────────── UI ───────────────────── */
  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return WillPopScope(
      onWillPop: () async {
        if (_processingActive) {
          await _promptCancelProcessing();
          return false;
        }
        await _suspendCameraForRouteChange();
        return true;
      },
      child: Stack(
        children: [
          Scaffold(
            backgroundColor: const Color(0xFFF7F9FB),
            extendBodyBehindAppBar: _recording,
            appBar: AppBar(
          elevation: 0,
          backgroundColor:
              _recording ? Colors.transparent : const Color(0xFFF7F9FB),
          foregroundColor: _recording ? Colors.white : Colors.black,
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
          leading: IconButton(
            icon: const Icon(Icons.arrow_back),
            onPressed: () async {
              await _suspendCameraForRouteChange();
              if (!mounted) return;
              context.pop();
            },
          ),
          actions: [
            IconButton(
              icon: const Icon(Icons.download_rounded),
              tooltip: 'Export 2D (preview)',
              onPressed: _exportLive2D,
            ),
            IconButton(
              tooltip: _showLiveSkeleton ? 'Hide skeleton' : 'Show skeleton',
              icon:
                  Icon(_showLiveSkeleton ? Icons.visibility : Icons.visibility_off),
              onPressed: () =>
                  setState(() => _showLiveSkeleton = !_showLiveSkeleton),
            ),
          ],
        ),
        body: _recording
            ? Stack(
                fit: StackFit.expand,
                children: [
                  _buildCameraContent(context, fullscreen: true),
                  SafeArea(
                    child: Align(
                      alignment: Alignment.bottomCenter,
                      child: Padding(
                        padding: const EdgeInsets.fromLTRB(16, 0, 16, 24),
                        child: _GradientButton(
                          height: 56,
                          onPressed: _toggleRecording,
                          colors: const [Color(0xFFD32F2F), Color(0xFFE53935)],
                          icon: Icons.stop,
                          label: 'Stop recording',
                        ),
                      ),
                    ),
                  ),
                ],
              )
            : Column(
                children: [
                  Expanded(
                    child: Padding(
                      padding: const EdgeInsets.fromLTRB(16, 16, 16, 0),
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(28),
                        clipBehavior: Clip.antiAlias,
                        child: _buildCameraContent(context, fullscreen: false),
                      ),
                    ),
                  ),
                  SafeArea(
                    top: false,
                    child: Padding(
                      padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
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
          ),
          if (_processingEvent.state != ProgressState.hidden)
            Positioned.fill(
              child: IgnorePointer(
                ignoring: !_processingActive,
                child: _ProcessingBanner(
                  event: _processingEvent,
                  onCancel:
                      _processingActive ? _handleProcessingCancel : null,
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildCameraContent(BuildContext context, {required bool fullscreen}) {
    if (!_ready || !_cam.value.isInitialized) {
      return const ColoredBox(
        color: Colors.black,
        child: Center(child: CircularProgressIndicator()),
      );
    }

    final previewSize = _cam.value.previewSize;
    if (previewSize == null) {
      return const ColoredBox(
        color: Colors.black,
        child: Center(child: CircularProgressIndicator()),
      );
    }

    final bool showLiveSkeleton = _showLiveSkeleton &&
        _latestPts != null &&
        _latestImgSize != null &&
        !_skeletonHiddenAfterMatch;

    final bool showOutline = fullscreen &&
        !_hideOutline &&
        _refOverlayPts != null &&
        _latestImgSize != null;

    final double topPadding = fullscreen
        ? MediaQuery.of(context).padding.top + 16
        : 16.0;

    Widget buildTiltIndicator() {
      return AnimatedBuilder(
        animation: _flashCtrl,
        builder: (_, __) {
          final bool outOfRange = _tiltTooLow || _tiltTooHigh;
          final double flashingOpacity =
              outOfRange ? (0.55 + 0.45 * _flashCtrl.value) : 1.0;

          final IconData icon = _tiltTooLow
              ? Icons.north_rounded
              : (_tiltTooHigh
                  ? Icons.south_rounded
                  : Icons.check_circle_rounded);

          final Color iconColor =
              (_tiltTooLow || _tiltTooHigh) ? Colors.orangeAccent : Colors.greenAccent;

          final String label = _tiltTooLow
              ? 'Increase tilt'
              : (_tiltTooHigh ? 'Decrease tilt' : 'Perfect');

          return Opacity(
            opacity: flashingOpacity,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(icon, size: 68, color: iconColor),
                const SizedBox(height: 12),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 12),
                  decoration: BoxDecoration(
                    color: Colors.black.withOpacity(.42),
                    borderRadius: BorderRadius.circular(18),
                  ),
                  child: Text(
                    label,
                    textAlign: TextAlign.center,
                    style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.w800,
                      fontSize: 22,
                      letterSpacing: .2,
                    ),
                  ),
                ),
              ],
            ),
          );
        },
      );
    }

    final Widget stack = Stack(
      fit: StackFit.expand,
      children: [
        CameraPreview(_cam),
        if (showLiveSkeleton)
          AnimatedBuilder(
            animation: _skeletonAnimation,
            builder: (context, _) {
              final opacity = (1.0 - _skeletonFadeCtrl.value).clamp(0.0, 1.0);
              if (opacity <= 0.0) {
                return const SizedBox.shrink();
              }
              final color = _skeletonColorAnim.value ?? Colors.redAccent;
              return Opacity(
                opacity: opacity,
                child: LiveSkeletonOverlay(
                  points: _latestPts,
                  imageSize: _latestImgSize!,
                  referencePoints: null,
                  mirrorHorizontally: false,
                  color: color,
                  thickness: 3.5,
                  showJoints: false,
                  boxFit: BoxFit.cover,
                  alignment: Alignment.center,
                ),
              );
            },
          ),
        Align(
          alignment: Alignment.topCenter,
          child: Padding(
            padding: EdgeInsets.only(top: topPadding),
            child: buildTiltIndicator(),
          ),
        ),
        if (showOutline)
          IgnorePointer(
            child: AnimatedOpacity(
              duration: const Duration(milliseconds: 350),
              opacity: _calibLockedGreen ? 0.0 : 1.0,
              child: AnimatedBuilder(
                animation: _glowCtrl,
                builder: (_, __) {
                  final color =
                      _calibLockedGreen ? Colors.greenAccent : Colors.redAccent;
                  return CustomPaint(
                    painter: _ReferenceSkeletonOutlinePainter(
                      points: _refOverlayPts!,
                      imageSize: _latestImgSize!,
                      mirror: false,
                      pulse: _glowCtrl.value,
                      color: color,
                      isGreen: _calibLockedGreen,
                      boxFit: BoxFit.cover,
                      alignment: Alignment.center,
                    ),
                  );
                },
              ),
            ),
          ),
        if (_calibLockedGreen && showOutline)
          CustomPaint(
            painter: _ReferenceSkeletonOutlinePainter(
              points: _refOverlayPts!,
              imageSize: _latestImgSize!,
              mirror: false,
              pulse: 1.0,
              color: Colors.greenAccent,
              isGreen: true,
              boxFit: BoxFit.cover,
              alignment: Alignment.center,
            ),
          ),
        const _BottomScrim(),
      ],
    );

    return ColoredBox(
      color: Colors.black,
      child: FittedBox(
        fit: BoxFit.cover,
        child: SizedBox(
          width: previewSize.height,
          height: previewSize.width,
          child: stack,
        ),
      ),
    );
  }

  Future<void> _toggleRecording() async {
    if (_recording) {
      // Stop recording → resume live stream
      _pendingRecording = null;
      await _stopRecordingSafely();
      await Future.delayed(const Duration(milliseconds: 50)); // let native drain
      setState(() => _recording = false);
      _logPoseEvents = false;
      final startedAt = _recordingStartedAt ?? DateTime.now();
      final endedAt = DateTime.now();
      _recordingStartedAt = null;
      final xFile = _pendingRecording;
      _pendingRecording = null;
      if (xFile != null) {
        final saved = await _rememberVideo(File(xFile.path));
        await _processRecordedSession(saved, startedAt, endedAt);
      }
      if (!_cam.value.isStreamingImages) {
        await Future.delayed(const Duration(milliseconds: 50));
        await _setStreaming(true);
      }
      HapticFeedback.selectionClick();
    } else {
      // Start recording → stop live stream (Camera cannot do both)
      if (_cam.value.isStreamingImages) {
        await _setStreaming(false);
        await Future.delayed(const Duration(milliseconds: 50)); // grace
      }
      await _cam.prepareForVideoRecording();
      _pendingRecording = null;
      await _cam.startVideoRecording();
      setState(() => _recording = true);
      _logPoseEvents = true;
      _recordingStartedAt = DateTime.now();
      HapticFeedback.heavyImpact();
    }
  }

  // POST { "s3_key": "<path>", … } to Flask /enqueue
  Future<void> _notifyBackend(String s3Path) async {
    const backendEndpoint = 'http://63.178.80.242:5001/enqueue';
    try {
      final sess = await amp.Amplify.Auth.fetchAuthSession() as CognitoAuthSession;
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
        padding: const EdgeInsets.fromLTRB(16, 14, 16, 16),
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

class _ProcessingBanner extends StatelessWidget {
  const _ProcessingBanner({required this.event, this.onCancel});

  final ProgressEvent event;
  final VoidCallback? onCancel;

  @override
  Widget build(BuildContext context) {
    if (event.state == ProgressState.hidden) {
      return const SizedBox.shrink();
    }
    final theme = Theme.of(context);
    final determinate = event.state == ProgressState.determinate;
    final value = determinate ? (event.value ?? 0.0).clamp(0.0, 1.0) : null;
    final subtitle = determinate && event.processed != null && event.total != null
        ? '${event.processed} of ${event.total} windows'
        : null;
    final canCancel =
        onCancel != null && event.state != ProgressState.complete && event.state != ProgressState.error;

    return SafeArea(
      child: Align(
        alignment: Alignment.topCenter,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 0),
          child: Material(
            elevation: 8,
            borderRadius: BorderRadius.circular(22),
            color: theme.colorScheme.surface,
            shadowColor: Colors.black.withOpacity(0.18),
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 18),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              event.state == ProgressState.complete
                                  ? 'Processing complete'
                                  : event.phase,
                              style: theme.textTheme.titleMedium?.copyWith(
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                            if (subtitle != null)
                              Padding(
                                padding: const EdgeInsets.only(top: 4),
                                child: Text(
                                  subtitle,
                                  style: theme.textTheme.bodySmall?.copyWith(
                                    color: theme.textTheme.bodySmall?.color?.withOpacity(0.7),
                                  ),
                                ),
                              ),
                          ],
                        ),
                      ),
                      IconButton(
                        onPressed: canCancel ? onCancel : null,
                        icon: Icon(
                          event.state == ProgressState.complete
                              ? Icons.check_circle_rounded
                              : Icons.close_rounded,
                        ),
                        color: canCancel
                            ? theme.colorScheme.primary
                            : theme.colorScheme.outline,
                        tooltip: canCancel ? 'Cancel processing' : null,
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(999),
                    child: SizedBox(
                      height: 8,
                      child: determinate
                          ? LinearProgressIndicator(value: value)
                          : const LinearProgressIndicator(),
                    ),
                  ),
                ],
              ),
            ),
          ),
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

/// Painter that reuses the reference skeleton coordinates and renders them as
/// a glowing outline. This keeps the silhouette perfectly aligned with the
/// reference pose that powers the matcher.
class _ReferenceSkeletonOutlinePainter extends CustomPainter {
  _ReferenceSkeletonOutlinePainter({
    required this.points,
    required this.imageSize,
    required this.mirror,
    required this.pulse,
    required this.color,
    required this.isGreen,
    required this.boxFit,
    required this.alignment,
  });

  final List<Offset> points;
  final Size imageSize;
  final bool mirror;
  final double pulse; // 0..1
  final Color color;
  final bool isGreen;
  final BoxFit boxFit;
  final Alignment alignment;

  @override
  void paint(Canvas canvas, Size size) {
    if (points.length < 17 || imageSize.width <= 0 || imageSize.height <= 0) {
      return;
    }

    final fitted = applyBoxFit(boxFit, imageSize, size);
    final dest = Size(fitted.destination.width, fitted.destination.height);
    if (dest.width <= 0 || dest.height <= 0) return;
    final scaleX = dest.width / imageSize.width;
    final scaleY = dest.height / imageSize.height;

    final fx = (alignment.x + 1) / 2.0;
    final fy = (alignment.y + 1) / 2.0;
    final offsetX = (size.width - dest.width) * fx;
    final offsetY = (size.height - dest.height) * fy;

    Offset mapPoint(Offset p) {
      double xLocal = p.dx * scaleX;
      double yLocal = p.dy * scaleY;
      if (mirror) {
        xLocal = dest.width - xLocal;
      }
      return Offset(offsetX + xLocal, offsetY + yLocal);
    }

    final mapped = List<Offset?>.filled(points.length, null);
    for (var i = 0; i < points.length; i++) {
      final p = points[i];
      if (_validPoint(p)) {
        mapped[i] = mapPoint(p);
      }
    }

    final double baseStroke = dest.shortestSide * 0.035;
    final double strokeWidth = isGreen ? baseStroke * 0.85 : baseStroke;
    final double glowWidth = strokeWidth +
        (isGreen ? baseStroke * 0.45 : baseStroke * (0.75 + 0.45 * pulse));
    final double glowAlpha = isGreen ? 0.92 : (0.72 + 0.28 * pulse);

    final Paint glow = Paint()
      ..color = color.withOpacity(glowAlpha)
      ..style = PaintingStyle.stroke
      ..strokeWidth = glowWidth
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round
      ..maskFilter = const MaskFilter.blur(BlurStyle.outer, 18);

    final Paint line = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;

    for (final edge in _edges) {
      final Offset? a = mapped[edge[0]];
      final Offset? b = mapped[edge[1]];
      if (a == null || b == null) continue;
      canvas.drawLine(a, b, glow);
    }

    for (final edge in _edges) {
      final Offset? a = mapped[edge[0]];
      final Offset? b = mapped[edge[1]];
      if (a == null || b == null) continue;
      canvas.drawLine(a, b, line);
    }
  }

  bool _validPoint(Offset p) =>
      p.dx.isFinite && p.dy.isFinite && p.dx != 0 && p.dy != 0;

  @override
  bool shouldRepaint(covariant _ReferenceSkeletonOutlinePainter oldDelegate) {
    return oldDelegate.points != points ||
        oldDelegate.imageSize != imageSize ||
        oldDelegate.mirror != mirror ||
        oldDelegate.pulse != pulse ||
        oldDelegate.color != color ||
        oldDelegate.isGreen != isGreen ||
        oldDelegate.boxFit != boxFit ||
        oldDelegate.alignment != alignment;
  }
}
