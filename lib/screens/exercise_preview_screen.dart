// ─────────────────────────────────────────────────────────────────────────────
// lib/screens/exercise_preview_screen.dart
// Streams live camera frames for pose estimation, logs JSONL, and runs MotionBERT.
// Includes setup & calibration (front cam tilt + 2D pose gating) with live overlay.
// ─────────────────────────────────────────────────────────────────────────────
import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:ui';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart' show Listenable, kDebugMode;
import 'package:flutter/material.dart';
import 'package:flutter/painting.dart' show applyBoxFit;
import 'package:go_router/go_router.dart';
import 'package:intl/intl.dart';
import 'package:path_provider/path_provider.dart';
import 'package:sensors_plus/sensors_plus.dart';

import '../shared/services/live_pose.dart'; // LIVE RTM-Pose 2D
import '../shared/services/pose_matcher.dart';
import '../shared/services/pose_runtime.dart'; // offline pipeline
import '../shared/widgets/live_skeleton_overlay.dart';

enum CaptureState { idle, streaming, stopping, processing3D }

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
  bool _mirrorPreview = true; // front camera previews are mirrored visually

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

  /* ───────────── capture/session state ───────────── */
  late final LivePosePreviewEngine _engine;
  final PosePipeline _pipeline = PosePipeline();
  CaptureState _state = CaptureState.idle;
  bool _runningEst = false; // simple reentrancy guard
  bool _poseOk = false; // true when detected 2D matches desired pose
  String? _sessionId;
  IOSink? _jsonlSink;
  File? _jsonlFile;
  DateTime? _captureStartedAt;
  DateTime? _captureStoppedAt;
  final List<Pose2DFrame> _capturedFrames = <Pose2DFrame>[];
  int _frameStride = 3; // drop frames when busy
  int _frameSkip = 0;
  bool _imageStreamActive = false;

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
    _engine = LivePosePreviewEngine(
      roiMargin: 1.25,
      confidenceThreshold: 0.15,
    );
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

    try {
      await _engine.ensureInitialized();
    } catch (e, stack) {
      if (kDebugMode) {
        debugPrint('[ExercisePreview] engine init error: $e\n$stack');
      }
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Pose engine failed to initialize: $e')),
        );
      }
    }

    // Wait for explicit start to begin streaming frames
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
      if (_imageStreamActive) return;
      await _cam.startImageStream((CameraImage img) {
        _onImageFromCamera(img); // fire-and-forget
      });
      _imageStreamActive = true;
    } else {
      if (!_imageStreamActive) return;
      try {
        await _cam.stopImageStream();
      } catch (_) {}
      _imageStreamActive = false;
      await Future.delayed(const Duration(milliseconds: 30)); // let native drain
    }
  }

  Future<void> _onImageFromCamera(CameraImage img) async {
    if (_state != CaptureState.streaming) return;
    if (_runningEst) return;
    _frameSkip = (_frameSkip + 1) % math.max(1, _frameStride);
    if (_frameSkip != 0) return;
    _runningEst = true;
    try {
      final frame = await _engine.processFrame(img);
      final pts = frame == null
          ? null
          : frame.kptsCoco
              .map((p) => Offset(p[0].toDouble(), p[1].toDouble()))
              .toList(growable: false);

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

      if (frame != null) {
        _recordPoseFrame(frame, img);
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

  void _recordPoseFrame(LivePoseFrame frame, CameraImage img) {
    if (_jsonlSink == null) return;
    final start = _captureStartedAt;
    final frameIndex = _capturedFrames.length;
    final timestamp = start == null
        ? frameIndex.toDouble()
        : DateTime.now().difference(start).inMicroseconds / 1e6;
    final keypoints = <PoseKeypoint2D>[];
    final coco = frame.kptsCoco;
    for (var i = 0; i < coco.length; i++) {
      final p = coco[i];
      keypoints.add(
        PoseKeypoint2D(
          id: i,
          x: p[0],
          y: p[1],
          c: p.length > 2 ? p[2] : 0.0,
        ),
      );
    }
    final poseFrame = Pose2DFrame(
      frameIndex: frameIndex,
      t: timestamp,
      imgW: img.width,
      imgH: img.height,
      keypoints: keypoints,
    );
    _capturedFrames.add(poseFrame);
    final line = {
      'frameIndex': poseFrame.frameIndex,
      't': double.parse(poseFrame.t.toStringAsFixed(4)),
      'imgW': poseFrame.imgW,
      'imgH': poseFrame.imgH,
      'confidence': frame.confMean,
      'bbox': frame.bbox,
      'keypoints': keypoints
          .map((k) => {'id': k.id, 'x': k.x, 'y': k.y, 'c': k.c ?? 0.0})
          .toList(growable: false),
    };
    _jsonlSink?.writeln(jsonEncode(line));
  }

  Future<void> _onStart() async {
    if (_state != CaptureState.idle) return;
    if (!_cam.value.isInitialized) return;

    final previousSink = _jsonlSink;
    if (previousSink != null) {
      await previousSink.flush();
      await previousSink.close();
    }

    final dir = await getTemporaryDirectory();
    final sessionId = DateFormat('yyyyMMdd_HHmmss').format(DateTime.now());
    final sessionDir = Directory('${dir.path}/sessions/$sessionId');
    await sessionDir.create(recursive: true);

    _jsonlFile = File('${sessionDir.path}/coco_2d.jsonl');
    _jsonlSink = _jsonlFile!.openWrite(mode: FileMode.writeOnlyAppend);
    _capturedFrames.clear();
    _frameSkip = 0;
    _captureStartedAt = DateTime.now();
    _captureStoppedAt = null;
    _sessionId = sessionId;

    setState(() => _state = CaptureState.streaming);
    await _setStreaming(true);
  }

  Future<void> _onStop() async {
    if (_state != CaptureState.streaming) return;
    setState(() => _state = CaptureState.stopping);

    await _setStreaming(false);
    _captureStoppedAt = DateTime.now();

    final sink = _jsonlSink;
    _jsonlSink = null;
    if (sink != null) {
      await sink.flush();
      await sink.close();
    }

    final file = _jsonlFile;
    if (file == null || _capturedFrames.isEmpty) {
      if (mounted) {
        setState(() => _state = CaptureState.idle);
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('No pose frames captured. Try again.')),
        );
      } else {
        _state = CaptureState.idle;
      }
      _jsonlFile = null;
      _capturedFrames.clear();
      return;
    }

    setState(() => _state = CaptureState.processing3D);

    try {
      final payload = await _runMotionBert(file.path);
      if (!mounted) return;
      context.go(
        '/feedback',
        extra: payload,
      );
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Motion analysis failed: $e')),
        );
      }
    } finally {
      _jsonlFile = null;
      _capturedFrames.clear();
      if (mounted) {
        setState(() => _state = CaptureState.idle);
      } else {
        _state = CaptureState.idle;
      }
    }
  }

  Future<Map<String, dynamic>> _runMotionBert(String jsonlPath) async {
    final frames = List<Pose2DFrame>.from(_capturedFrames);
    final fps = _estimateFps();
    final seq = Pose2DSequence(
      frames: frames,
      fps: fps,
      imageWidth: frames.isEmpty ? 0 : frames.first.imgW,
      imageHeight: frames.isEmpty ? 0 : frames.first.imgH,
    );
    final result3d = await _pipeline.estimate3D(seq);
    return {
      'sessionId': _sessionId,
      'exerciseId': widget.exerciseId,
      'jsonlPath': jsonlPath,
      'pose2d': seq.frames.map((f) => f.toJson()).toList(growable: false),
      'pose3d': result3d.sequence,
      'windows': result3d.windows.map((w) => w.toJson()).toList(growable: false),
      'windowSize': result3d.windowSize,
      'stride': result3d.stride,
      'fps': fps,
    };
  }

  double _estimateFps() {
    final start = _captureStartedAt;
    final stop = _captureStoppedAt ?? DateTime.now();
    if (start == null) {
      return 10.0;
    }
    final durationMs = stop.difference(start).inMilliseconds;
    if (durationMs <= 0) {
      return 10.0;
    }
    final seconds = durationMs / 1000.0;
    if (_capturedFrames.isEmpty) {
      return 0.0;
    }
    return _capturedFrames.length / seconds;
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

  /* ─────────────────── CAMERA ─────────────────── */
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
    _flashCtrl.dispose();
    _glowCtrl.dispose();
    _skeletonColorCtrl.dispose();
    _skeletonFadeCtrl.dispose();
    _logPoseEvents = false;
    if (_imageStreamActive) {
      try {
        _cam.stopImageStream();
      } catch (_) {}
    }
    final sink = _jsonlSink;
    _jsonlSink = null;
    if (sink != null) {
      // ignore: discarded_futures
      sink.close();
    }
    _engine.dispose();
    _cam.dispose();
    super.dispose();
  }

  /* ───────────────────── UI ───────────────────── */
  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isStreaming = _state == CaptureState.streaming;
    final isProcessing = _state == CaptureState.processing3D;

    return Scaffold(
      backgroundColor: const Color(0xFFF7F9FB),
      appBar: AppBar(
        elevation: 0,
        backgroundColor: const Color(0xFFF7F9FB),
        foregroundColor: Colors.black,
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
            if (_state == CaptureState.streaming) {
              await _setStreaming(false);
            }
            final sink = _jsonlSink;
            if (sink != null) {
              await sink.flush();
              await sink.close();
            }
            _jsonlSink = null;
            _jsonlFile = null;
            _capturedFrames.clear();
            _sessionId = null;
            _captureStartedAt = null;
            _captureStoppedAt = null;
            if (mounted) {
              setState(() {
                _state = CaptureState.idle;
              });
              context.pop();
            }
          },
        ),
        actions: [
          IconButton(
            tooltip: _showLiveSkeleton ? 'Hide skeleton' : 'Show skeleton',
            icon: Icon(_showLiveSkeleton ? Icons.visibility : Icons.visibility_off),
            onPressed: () => setState(() => _showLiveSkeleton = !_showLiveSkeleton),
          ),
        ],
      ),
      body: Stack(
        children: [
          Column(
            children: [
              Expanded(
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(16, 16, 16, 0),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(28),
                    clipBehavior: Clip.antiAlias,
                    child: _buildCameraContent(context, fullscreen: true),
                  ),
                ),
              ),
              SafeArea(
                top: false,
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(16, 12, 16, 24),
                  child: _buildControlBar(isStreaming),
                ),
              ),
            ],
          ),
          if (isProcessing)
            Positioned.fill(
              child: _buildProcessingOverlay(),
            ),
        ],
      ),
    );
  }

  Widget _buildControlBar(bool isStreaming) {
    final bool canStart = _state == CaptureState.idle && _ready;
    final bool canStop = isStreaming;
    return Row(
      children: [
        Expanded(
          child: ElevatedButton.icon(
            onPressed: canStart ? _onStart : null,
            icon: const Icon(Icons.play_arrow_rounded),
            label: const Text('Start capture'),
            style: ElevatedButton.styleFrom(
              minimumSize: const Size.fromHeight(52),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
            ),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: ElevatedButton.icon(
            onPressed: canStop ? _onStop : null,
            icon: const Icon(Icons.stop_rounded),
            label: const Text('Stop'),
            style: ElevatedButton.styleFrom(
              minimumSize: const Size.fromHeight(52),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
              backgroundColor: Colors.redAccent,
              foregroundColor: Colors.white,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildProcessingOverlay() {
    return Container(
      color: Colors.black.withOpacity(0.6),
      child: const Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            CircularProgressIndicator(valueColor: AlwaysStoppedAnimation<Color>(Colors.white)),
            SizedBox(height: 16),
            Text(
              'Running MotionBERT…',
              style: TextStyle(
                color: Colors.white,
                fontSize: 16,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
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

// COCO-17 skeleton connectivity (indices match your 17-point order):
// 0:nose 1:lEye 2:rEye 3:lEar 4:rEar 5:lSh 6:rSh 7:lElb 8:rElb
// 9:lWrist 10:rWrist 11:lHip 12:rHip 13:lKnee 14:rKnee 15:lAnk 16:rAnk
const List<List<int>> _edges = <List<int>>[
  [5, 6],   // shoulders
  [5, 7],   // left upper arm
  [7, 9],   // left forearm
  [6, 8],   // right upper arm
  [8, 10],  // right forearm
  [11, 12], // hips
  [5, 11],  // left torso
  [6, 12],  // right torso
  [11, 13], // left thigh
  [13, 15], // left shin
  [12, 14], // right thigh
  [14, 16], // right shin
  [0, 5],   // nose to left shoulder (optional head/neck link)
  [0, 6],   // nose to right shoulder
  // (You can add eye/ear links if you want the head triangle,
  // e.g., [1,0], [2,0], [1,3], [2,4])
];


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
