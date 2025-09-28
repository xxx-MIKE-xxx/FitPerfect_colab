import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:flutter/painting.dart'; // for applyBoxFit, Alignment math

/// Draws a COCO-17 skeleton (live + optional reference).
/// Expects BOTH `points` and `referencePoints` to be in the SAME raw camera space.
/// This widget maps raw-camera coordinates to the widget using a configurable BoxFit
/// (default: BoxFit.contain) and centers them correctly, accounting for letterboxing.
///
/// Tip: If you align your reference to the live frame first (e.g. with
/// PoseMatcher.referenceAlignedToLive(livePoints)), you can pass it here as
/// `referencePoints` and it will render exactly on top in the correct position.
class LiveSkeletonOverlay extends StatelessWidget {
  const LiveSkeletonOverlay({
    super.key,
    required this.points,            // live points in RAW camera space
    required this.imageSize,         // raw camera size (w,h)
    this.referencePoints,            // aligned reference points in SAME space
    this.mirrorHorizontally = false, // set true for front camera preview
    this.color = Colors.cyanAccent,
    this.thickness = 3.0,
    this.showJoints = true,
    this.jointRadius = 3.5,
    this.refColor = Colors.pinkAccent,
    this.refThickness = 2.5,
    this.showRefJoints = true,

    /// New: match how your camera preview is laid out.
    /// If your preview uses FittedBox(BoxFit.contain), leave defaults.
    /// If it uses BoxFit.cover, set boxFit: BoxFit.cover.
    this.boxFit = BoxFit.contain,
    this.alignment = Alignment.center,
  });

  final List<Offset>? points;
  final Size imageSize;

  final List<Offset>? referencePoints;

  final bool mirrorHorizontally;

  final Color color;
  final double thickness;
  final bool showJoints;
  final double jointRadius;

  final Color refColor;
  final double refThickness;
  final bool showRefJoints;

  /// How to fit the raw camera frame into this widget's size.
  final BoxFit boxFit;

  /// Where to place the fitted image inside this widget (used with letterboxing).
  final Alignment alignment;

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _SkelPainter(
        ptsLive: points,
        ptsRef: referencePoints,
        imageSize: imageSize,
        mirror: mirrorHorizontally,
        liveColor: color,
        liveT: thickness,
        showLiveJoints: showJoints,
        jointR: jointRadius,
        refColor: refColor.withOpacity(0.90),
        refT: refThickness,
        showRefJoints: showRefJoints,
        boxFit: boxFit,
        alignment: alignment,
      ),
    );
  }
}

// COCO-17 edges (by joint index)
const _edges = <List<int>>[
  [5, 6],  // shoulders
  [5, 7], [7, 9], // left arm
  [6, 8], [8,10], // right arm
  [11,12], // hips
  [5,11], [6,12], // torso links
  [11,13], [13,15], // left leg
  [12,14], [14,16], // right leg
  [0,5], [0,6],     // head to shoulders
];

class _SkelPainter extends CustomPainter {
  _SkelPainter({
    required this.ptsLive,
    required this.ptsRef,
    required this.imageSize,
    required this.mirror,
    required this.liveColor,
    required this.liveT,
    required this.showLiveJoints,
    required this.jointR,
    required this.refColor,
    required this.refT,
    required this.showRefJoints,
    required this.boxFit,
    required this.alignment,
  });

  final List<Offset>? ptsLive;
  final List<Offset>? ptsRef;
  final Size imageSize;
  final bool mirror;
  final Color liveColor;
  final double liveT;
  final bool showLiveJoints;
  final double jointR;

  final Color refColor;
  final double refT;
  final bool showRefJoints;

  final BoxFit boxFit;
  final Alignment alignment;

  @override
  void paint(Canvas canvas, Size size) {
    if (imageSize.width <= 0 || imageSize.height <= 0) return;

    // Compute fitted destination rect for the raw camera image inside this widget.
    // This mirrors FittedBox + BoxFit layout so overlay stays perfectly aligned.
    final fitted = applyBoxFit(boxFit, imageSize, size);
    final dest = Size(fitted.destination.width, fitted.destination.height);
    final scaleX = dest.width / imageSize.width;
    final scaleY = dest.height / imageSize.height;

    // Position the dest rect inside the widget according to alignment.
    // alignment.x/y are in [-1,1]; convert to [0,1] to get the inset factor.
    final fx = (alignment.x + 1) / 2.0;
    final fy = (alignment.y + 1) / 2.0;
    final offsetX = (size.width - dest.width) * fx;
    final offsetY = (size.height - dest.height) * fy;

    Offset map(Offset p) {
      // Map to local coords within the fitted dest rect first,
      // then apply mirroring within that rect (not the whole widget),
      // finally add the dest offset.
      double xLocal = p.dx * scaleX;
      double yLocal = p.dy * scaleY;

      if (mirror) {
        xLocal = dest.width - xLocal; // flip around the fitted rect
      }

      return Offset(offsetX + xLocal, offsetY + yLocal);
    }

    // Draw reference first (behind, thinner)
    if (ptsRef != null && ptsRef!.length >= 17) {
      final refPaint = Paint()
        ..color = refColor
        ..style = PaintingStyle.stroke
        ..strokeWidth = refT
        ..strokeCap = StrokeCap.round
        ..strokeJoin = StrokeJoin.round;

      for (final e in _edges) {
        final a = ptsRef![e[0]];
        final b = ptsRef![e[1]];
        if (!_valid(a) || !_valid(b)) continue;
        canvas.drawLine(map(a), map(b), refPaint);
      }
      if (showRefJoints) {
        final jPaint = Paint()..color = refColor;
        for (final p in ptsRef!) {
          if (!_valid(p)) continue;
          canvas.drawCircle(map(p), jointR * 0.9, jPaint);
        }
      }
    }

    // Draw live (on top, thicker)
    if (ptsLive != null && ptsLive!.length >= 17) {
      final livePaint = Paint()
        ..color = liveColor
        ..style = PaintingStyle.stroke
        ..strokeWidth = liveT
        ..strokeCap = StrokeCap.round
        ..strokeJoin = StrokeJoin.round;

      for (final e in _edges) {
        final a = ptsLive![e[0]];
        final b = ptsLive![e[1]];
        if (!_valid(a) || !_valid(b)) continue;
        canvas.drawLine(map(a), map(b), livePaint);
      }
      if (showLiveJoints) {
        final jPaint = Paint()..color = liveColor;
        for (final p in ptsLive!) {
          if (!_valid(p)) continue;
          canvas.drawCircle(map(p), jointR, jPaint);
        }
      }
    }
  }

  bool _valid(Offset p) =>
      p.dx.isFinite && p.dy.isFinite && p.dx != 0 && p.dy != 0;

  @override
  bool shouldRepaint(covariant _SkelPainter old) {
    return old.ptsLive != ptsLive ||
        old.ptsRef != ptsRef ||
        old.imageSize != imageSize ||
        old.mirror != mirror ||
        old.liveColor != liveColor ||
        old.refColor != refColor ||
        old.liveT != liveT ||
        old.refT != refT ||
        old.showLiveJoints != showLiveJoints ||
        old.showRefJoints != showRefJoints ||
        old.boxFit != boxFit ||
        old.alignment != alignment;
  }
}
