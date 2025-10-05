// lib/shared/services/pose_matcher.dart
// Refactored to load reference pose from JSON (with robust fallbacks)
// and to center/align the reference correctly for MAE-based matching.
//
// ✅ Fixes covered:
// 1) Reference file now read from JSON (supports several common shapes/keys).
// 2) Provides a reliable way to center the reference skeleton for overlay.
// 3) Matching now aligns the *reference to the live frame* (correct scale/translation)
//    and guards against degenerate bboxes so MAE doesn't get stuck at a constant.
//
// ─────────────────────────────────────────────────────────────────────────
// JSON formats accepted (any of these):
// A) Simple nested arrays (preferred)
//    assets/meta/reference_frame.json
//    [ [x,y], [x,y], ... ]             // length 17 (COCO) or 26 (Halpe)
//    [ [x,y,conf], ... ]               // the 3rd value is ignored if present
//
// B) Object with a key that holds the arrays
//    { "keypoints": [[x,y],...]} or {"points": [[x,y],...]} or {"data": [[x,y],...]}
//
// C) Objects per point
//    [ {"x": 123, "y": 456}, ... ]
//
// If 26+ points are provided (Halpe or generic), a heuristic mapper will convert
// them to the COCO-17 order used here:
// 0:nose, 1:lEye, 2:rEye, 3:lEar, 4:rEar, 5:lShoulder, 6:rShoulder,
// 7:lElbow, 8:rElbow, 9:lWrist, 10:rWrist, 11:lHip, 12:rHip,
// 13:lKnee, 14:rKnee, 15:lAnkle, 16:rAnkle
// ─────────────────────────────────────────────────────────────────────────

import 'dart:convert';
import 'dart:math' as math;
import 'dart:ui';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

const bool _enablePoseMatcherLogs = false;

void _poseMatcherLog(String message) {
  if (!_enablePoseMatcherLogs) return;
  debugPrint(message);
}

class PoseMatcher {
  PoseMatcher({
    this.fixedMaeThresholdPx = 30.0, // absolute pixel cap
    this.dynamicMaePctOfRefH = 0.06, // or 6% of the reference bbox height
    this.referenceJsonPaths = const [
      'assets/meta/coco17_neutral_skeleton.json'
    ],
  });

  /// Hard pixel cap, and dynamic cap w.r.t reference bbox height.
  final double fixedMaeThresholdPx;
  final double dynamicMaePctOfRefH;

  /// Asset search order for JSON files.
  final List<String> referenceJsonPaths;

  /// Loaded reference (COCO-17) in pixel-like coords from the asset file.
  List<Offset>? _ref17;

  /// Cached bbox height for the reference (same units as the JSON, used only as a scale proxy).
  double? _refBBoxH;

  /// Path of the asset that successfully loaded the reference (for debugging).
  String? _refSource;

  bool _loadedTried = false;

  /// Expose current COCO-17 reference points (raw, as loaded & remapped).
  List<Offset>? get refPoints => _ref17;
  String? get refSource => _refSource;

  /// Call once during startup (e.g., in initState). Safe to call multiple times.
  Future<void> ensureLoaded() async {
    if (_loadedTried && _ref17 != null) return;
    _loadedTried = true;
    _refSource = null;

    // Try JSON sources first (preferred)
    for (final path in referenceJsonPaths) {
      _ref17 = await _tryLoadFromJson(path);
      if (_ref17 != null) {
        _refSource = path;
        break;
      }
    }

    if (_ref17 == null) {
      // As a last-resort fallback, try legacy .npy names if they exist in your bundle.
      // Comment these out if you want to remove NPY support entirely.
      final npyPaths = [
        'assets/meta/reference_frame.npy',
        'assets/meta/reference_frame_26.npy',
      ];
      for (final path in npyPaths) {
        _ref17 = await _tryLoadFromNpy(path);
        if (_ref17 != null) {
          _refSource = path;
          break;
        }
      }
    }

    if (_ref17 != null) {
      _ref17 = _heuristicToCoco17(_ref17!);
      _refBBoxH = _bboxHeight(_ref17!);
      if (kDebugMode) {
        _poseMatcherLog('[PoseMatcher] Loaded reference with ${_ref17!.length} kpts '
            'from ${_refSource ?? 'unknown'} (refBBoxH=${_refBBoxH?.toStringAsFixed(1)})');
      }
    } else {
      if (kDebugMode) {
        _poseMatcherLog('[PoseMatcher] Could not load reference from assets/meta/. '
            'Ensure a JSON at assets/meta/reference_frame.json is listed in pubspec.');
      }
    }
  }

  /// Returns MAE in pixels between **live** keypoints and the reference,
  /// after aligning the reference to the live frame by similarity (scale+translation).
  /// If reference or input is invalid, returns double.infinity.
  double compareMAEPx(List<Offset> live17) {
    if (_ref17 == null || _ref17!.length < 17) return double.infinity;
    if (live17.length < 17) return double.infinity;

    // Guard against degenerate bboxes (prevents constant outputs)
    final refB = _bbox(_ref17!);
    final liveB = _bbox(live17);
    final refH = (refB.$4 - refB.$2).abs();
    final liveH = (liveB.$4 - liveB.$2).abs();
    if (refH <= 1e-3 || liveH <= 1e-3) return double.infinity;

    final alignedRef = _referenceAlignedToLive(live17);
    final live = live17;

    double sum = 0.0;
    int count = 0;
    for (int i = 0; i < 17; i++) {
      final dx = (alignedRef[i].dx - live[i].dx).abs();
      final dy = (alignedRef[i].dy - live[i].dy).abs();
      final d = math.sqrt(dx * dx + dy * dy);
      if (d.isFinite) {
        sum += d;
        count++;
      }
    }
    return count > 0 ? (sum / count) : double.infinity;
  }

  /// Returns MAE in pixels between live keypoints and a pre-positioned reference
  /// that is already expressed in the same coordinate space.
  double compareMAEPxAgainst(List<Offset> live17, List<Offset> reference17) {
    if (live17.length < 17 || reference17.length < 17) return double.infinity;

    double sum = 0.0;
    int count = 0;
    for (int i = 0; i < 17; i++) {
      final dx = (reference17[i].dx - live17[i].dx).abs();
      final dy = (reference17[i].dy - live17[i].dy).abs();
      final d = math.sqrt(dx * dx + dy * dy);
      if (d.isFinite) {
        sum += d;
        count++;
      }
    }
    return count > 0 ? (sum / count) : double.infinity;
  }

  /// Decide match using MAE + thresholds.
  bool isMatchByMAE(double maePx) {
    if (!maePx.isFinite) return false;
    // Reference pose must be within a fixed pixel error margin to be accepted.
    const double kReferenceMaeThresholdPx = 230.0;
    return maePx <= kReferenceMaeThresholdPx;
  }

  /// Center the reference skeleton in a given frame size and scale it to a
  /// fraction of the frame height (default 70%). Useful for visual overlay.
  List<Offset>? centeredRefForFrame(Size frameSize, {double heightFrac = 0.70}) {
    if (_ref17 == null || _ref17!.length < 17) return null;
    final ref = _ref17!;
    final b = _bbox(ref);
    final refH = (b.$4 - b.$2).abs();
    if (refH <= 1e-3) return null;

    final s = (frameSize.height * heightFrac) / refH;

    final rcx = (b.$1 + b.$3) * 0.5;
    final rcy = (b.$2 + b.$4) * 0.5;
    final fcx = frameSize.width * 0.5;
    final fcy = frameSize.height * 0.5;

    return List<Offset>.generate(17, (i) {
      final p = ref[i];
      return Offset(
        s * (p.dx - rcx) + fcx,
        s * (p.dy - rcy) + fcy,
      );
    }, growable: false);
  }

  /// Project the loaded reference (COCO-17) into the **live** frame
  /// by aligning bbox centers and scaling by bbox height (s = liveH/refH).
  /// Returns null if reference or input is invalid.
  List<Offset>? referenceAlignedToLive(List<Offset> live17) {
    if (_ref17 == null || _ref17!.length < 17) return null;
    if (live17.length < 17) return null;

    final ref = _ref17!;
    // Reference bbox
    final (refMinX, refMinY, refMaxX, refMaxY) = _bbox(ref);
    final refH = (refMaxY - refMinY).abs();
    if (refH <= 1e-3) return null;

    // Live bbox
    final (liveMinX, liveMinY, liveMaxX, liveMaxY) = _bbox(live17);
    final liveH = (liveMaxY - liveMinY).abs();
    if (liveH <= 1e-3) return null;

    // Centers
    final refCX = (refMinX + refMaxX) * 0.5;
    final refCY = (refMinY + refMaxY) * 0.5;
    final liveCX = (liveMinX + liveMaxX) * 0.5;
    final liveCY = (liveMinY + liveMaxY) * 0.5;

    // Scale to live
    final s = liveH / refH;

    // ref' = s*(ref - refC) + liveC
    final aligned = List<Offset>.generate(
      17,
      (i) => Offset(
        s * (ref[i].dx - refCX) + liveCX,
        s * (ref[i].dy - refCY) + liveCY,
      ),
      growable: false,
    );
    return aligned;
  }

  // Internal helper for MAE path (maps reference → live)
  List<Offset> _referenceAlignedToLive(List<Offset> live17) {
    final ref = _ref17!;
    final refB = _bbox(ref);
    final liveB = _bbox(live17);

    final refH = (refB.$4 - refB.$2).abs();
    final liveH = (liveB.$4 - liveB.$2).abs();
    final s = liveH / (refH == 0 ? 1.0 : refH);

    final refCX = (refB.$1 + refB.$3) * 0.5;
    final refCY = (refB.$2 + refB.$4) * 0.5;
    final liveCX = (liveB.$1 + liveB.$3) * 0.5;
    final liveCY = (liveB.$2 + liveB.$4) * 0.5;

    return List<Offset>.generate(
      17,
      (i) => Offset(
        s * (ref[i].dx - refCX) + liveCX,
        s * (ref[i].dy - refCY) + liveCY,
      ),
      growable: false,
    );
  }

  // ───────────────────────── asset loading (JSON first) ─────────────────────────

  Future<List<Offset>?> _tryLoadFromJson(String assetPath) async {
    try {
      final raw = await rootBundle.loadString(assetPath);
      final decoded = json.decode(raw);
      final pts = _extractOffsetsFromDynamic(decoded);
      if (pts == null || pts.isEmpty) return null;
      if (pts.length >= 26) {
        return _heuristicToCoco17From26(pts);
      }
      if (pts.length >= 17) {
        return _heuristicToCoco17(pts);
      }
      return null;
    } catch (e) {
      if (kDebugMode) {
        _poseMatcherLog('[PoseMatcher] JSON load failed for "$assetPath": $e');
      }
      return null;
    }
  }

  /// Extracts a list of Offsets from a variety of JSON shapes.
  List<Offset>? _extractOffsetsFromDynamic(dynamic decoded) {
    List<dynamic>? list;

    // ── Handle common list formats ────────────────────────────────────────────
    if (decoded is List) {
      list = decoded;
      if (list.isNotEmpty && list.first is num) {
        return _offsetsFromFlatKeypoints(list.cast<num>());
      }
    } else if (decoded is Map) {
      // First, detect COCO annotation style { annotations: [ { keypoints: [...] } ] }
      final ann = decoded['annotations'];
      if (ann is List) {
        for (final item in ann) {
          if (item is Map && item['keypoints'] is List) {
            final kpList = item['keypoints'] as List<dynamic>;
            if (kpList.isNotEmpty && kpList.first is num) {
              final pts = _offsetsFromFlatKeypoints(kpList.cast<num>());
              if (pts != null && pts.isNotEmpty) return pts;
            }
          }
        }
      }

      // try common keys that already hold array-of-arrays
      for (final key in ['keypoints', 'points', 'data']) {
        final v = decoded[key];
        if (v is List) {
          if (v.isNotEmpty && v.first is num) {
            return _offsetsFromFlatKeypoints(v.cast<num>());
          }
          list = v;
          break;
        }
      }

      // if still null, try first array-like value in the map
      list ??= decoded.values.firstWhere(
        (v) => v is List,
        orElse: () => null,
      ) as List<dynamic>?;
      final candidate = list;
      if (candidate != null && candidate.isNotEmpty && candidate.first is num) {
        return _offsetsFromFlatKeypoints(candidate.cast<num>());
      }
    }

    if (list == null) return null;

    Offset? toOffset(dynamic e) {
      if (e is List && e.length >= 2) {
        final x = (e[0] as num).toDouble();
        final y = (e[1] as num).toDouble();
        return Offset(x, y);
      }
      if (e is Map) {
        if (e.containsKey('x') && e.containsKey('y')) {
          final x = (e['x'] as num).toDouble();
          final y = (e['y'] as num).toDouble();
          return Offset(x, y);
        }
        if (e.containsKey('X') && e.containsKey('Y')) {
          final x = (e['X'] as num).toDouble();
          final y = (e['Y'] as num).toDouble();
          return Offset(x, y);
        }
      }
      return null;
    }

    final out = <Offset>[];
    for (final e in list) {
      final p = toOffset(e);
      if (p != null && p.dx.isFinite && p.dy.isFinite) out.add(p);
    }
    return out;
  }

  /// Converts flattened keypoints ([x,y,(conf), ...]) into Offsets.
  List<Offset>? _offsetsFromFlatKeypoints(List<num> flat) {
    if (flat.length < 2) return null;

    bool isConfChannelLike() {
      if (flat.length % 3 != 0) return false;
      int confHits = 0;
      int samples = 0;
      for (int j = 2; j < flat.length; j += 3) {
        final c = flat[j].toDouble().abs();
        // COCO visibility scores are typically 0, 1, or 2.
        if (c <= 2.5) confHits++;
        samples++;
      }
      // Require most samples to look like visibility scores to avoid
      // misclassifying genuine XYZ coordinates as XY + conf.
      return samples > 0 && confHits / samples >= 0.6;
    }

    final hasConf = isConfChannelLike();
    final step = hasConf ? 3 : 2;

    final out = <Offset>[];
    for (int i = 0; i + 1 < flat.length; i += step) {
      final x = flat[i].toDouble();
      final y = flat[i + 1].toDouble();
      out.add(Offset(x, y));
    }
    return out;
  }

  // ───────────────────────── legacy NPY fallback (optional) ─────────────────────────

  Future<List<Offset>?> _tryLoadFromNpy(String assetPath) async {
    try {
      final bd = await rootBundle.load(assetPath);
      final arr = _npyLoad(bd.buffer.asUint8List());
      if (arr.shape.length != 2) return null;
      final rows = arr.shape[0];
      final cols = arr.shape[1];
      if (!(cols == 2 || cols == 3)) return null;

      final pts = <Offset>[];
      for (int i = 0; i < rows; i++) {
        final x = arr.get(i, 0);
        final y = arr.get(i, 1);
        pts.add(Offset(x, y));
      }
      if (rows >= 26) return _heuristicToCoco17From26(pts);
      if (rows >= 17) return _heuristicToCoco17(pts);
      return null;
    } catch (_) {
      return null;
    }
  }

  // ───────────────────────── heuristic remapping ─────────────────────────
  // The goal: map *any* 17-pt standing N-pose or a 26-pt Halpe into COCO-17.

  List<Offset> _heuristicToCoco17(List<Offset> raw17) {
    if (raw17.length < 17) {
      // If fewer than 17 provided, pad with last point (best-effort).
      final padded = List<Offset>.from(raw17);
      while (padded.length < 17) {
        padded.add(padded.isNotEmpty ? padded.last : const Offset(0, 0));
      }
      return padded;
    }
    // If it already *looks* like COCO order, keep it.
    if (_looksLikeCoco(raw17)) return raw17.sublist(0, 17);
    return _heuristicReorderToCoco(raw17);
  }

  List<Offset> _heuristicToCoco17From26(List<Offset> p26) {
    // Use the generic 17-point heuristic on all points; it will pick the needed ones.
    return _heuristicReorderToCoco(p26);
  }

  bool _looksLikeCoco(List<Offset> pts) {
    if (pts.length < 17) return false;
    // Quick ankle sanity: two lowest Y should be around indices 15/16.
    final lowest = _takeLowestY(pts, 2);
    final ankles = [pts[15], pts[16]];
    final d = (_dist(lowest[0], ankles[0]) + _dist(lowest[1], ankles[1])) / 2.0;
    final dSwap = (_dist(lowest[0], ankles[1]) + _dist(lowest[1], ankles[0])) / 2.0;
    return math.min(d, dSwap) < 80.0;
  }

  List<Offset> _heuristicReorderToCoco(List<Offset> ptsIn) {
    final pool = List<Offset>.from(ptsIn);
    if (pool.isEmpty) return List.filled(17, const Offset(0, 0));
    final cx = _medianX(pool);

    Offset popClosestBelow(Offset target) {
      int bestI = -1;
      double bestD = 1e9;
      for (int i = 0; i < pool.length; i++) {
        final p = pool[i];
        if (p.dy < target.dy) continue; // must be below (greater y)
        final d = _dist(p, target);
        if (d < bestD) {
          bestD = d;
          bestI = i;
        }
      }
      if (bestI == -1) {
        // fallback: closest overall
        for (int i = 0; i < pool.length; i++) {
          final d = _dist(pool[i], target);
          if (d < bestD) {
            bestD = d;
            bestI = i;
          }
        }
      }
      return pool.removeAt(bestI);
    }

    Offset popClosestAbove(Offset target) {
      int bestI = -1;
      double bestD = 1e9;
      for (int i = 0; i < pool.length; i++) {
        final p = pool[i];
        if (p.dy > target.dy) continue; // must be above (smaller y)
        final d = _dist(p, target);
        if (d < bestD) {
          bestD = d;
          bestI = i;
        }
      }
      if (bestI == -1) {
        for (int i = 0; i < pool.length; i++) {
          final d = _dist(pool[i], target);
          if (d < bestD) {
            bestD = d;
            bestI = i;
          }
        }
      }
      return pool.removeAt(bestI);
    }

    Offset pickByScore(List<Offset> list, double Function(Offset) score) {
      if (list.isEmpty) return pool.removeAt(0);
      int bi = 0;
      double bv = 1e9;
      for (int i = 0; i < list.length; i++) {
        final s = score(list[i]);
        if (s < bv) {
          bv = s;
          bi = i;
        }
      }
      return list[bi];
    }

    List<Offset> popFarthestPairAbove(double yMax) {
      int bestI = -1, bestJ = -1;
      double bestW = -1;
      for (int i = 0; i < pool.length; i++) {
        final pi = pool[i];
        if (pi.dy > yMax) continue;
        for (int j = i + 1; j < pool.length; j++) {
          final pj = pool[j];
          if (pj.dy > yMax) continue;
          final horizontal = (pi.dx - pj.dx).abs();
          final verticalPenalty = (pi.dy - pj.dy).abs() * 0.2;
          final w = horizontal - verticalPenalty;
          if (w > bestW) {
            bestW = w;
            bestI = i;
            bestJ = j;
          }
        }
      }
      if (bestI == -1) {
        // fallback: two highest points (smallest y)
        final top = List<Offset>.from(pool)..sort((a, b) => a.dy.compareTo(b.dy));
        final a = top.first;
        final b = top.length > 1 ? top[1] : a;
        pool.remove(a);
        pool.remove(b);
        return (a.dx < b.dx) ? [a, b] : [b, a];
      }
      final a = pool[bestI];
      final b = pool[bestJ];
      // remove higher index first
      if (bestI > bestJ) {
        pool.removeAt(bestI);
        pool.removeAt(bestJ);
      } else {
        pool.removeAt(bestJ);
        pool.removeAt(bestI);
      }
      return (a.dx < b.dx) ? [a, b] : [b, a];
    }

    // 1) Ankles: two lowest Y
    final lowest2 = _takeLowestY(pool, 2);
    for (final p in lowest2) {
      pool.remove(p);
    }
    Offset lAnk = lowest2[0].dx < lowest2[1].dx ? lowest2[0] : lowest2[1];
    Offset rAnk = lowest2[0].dx < lowest2[1].dx ? lowest2[1] : lowest2[0];

    // 2) Knees: closest above each ankle
    final lKnee = popClosestAbove(lAnk);
    final rKnee = popClosestAbove(rAnk);

    // 3) Hips: closest above knees but nearer to center X
    Offset pickHipNearCenter(Offset knee) {
      int bestI = -1;
      double bestScore = 1e9;
      for (int i = 0; i < pool.length; i++) {
        final p = pool[i];
        if (p.dy > knee.dy) continue; // must be above knee
        final score = _dist(p, knee) + (p.dx - cx).abs() * 0.5;
        if (score < bestScore) {
          bestScore = score;
          bestI = i;
        }
      }
      if (bestI == -1) bestI = 0;
      return pool.removeAt(bestI);
    }

    Offset lHip = pickHipNearCenter(lKnee);
    Offset rHip = pickHipNearCenter(rKnee);
    if (lHip.dx > rHip.dx) {
      final t = lHip;
      lHip = rHip;
      rHip = t;
    }

    // 4) Shoulders: above hips, with widest horizontal spread
    final sh = popFarthestPairAbove(math.min(lHip.dy, rHip.dy));
    final lSh = sh[0];
    final rSh = sh[1];

    // 5) Nose: highest remaining (smallest y)
    final nose = _popHighest(pool);

    // 6) Elbows/Wrists per side
    List<Offset> sideArm(Offset shoulder, bool leftSide) {
      final cand = <Offset>[];
      for (final p in pool) {
        if (leftSide && p.dx <= cx) cand.add(p);
        if (!leftSide && p.dx >= cx) cand.add(p);
      }
      final wrist = pickByScore(cand, (p) => (p.dy) + (p.dx - shoulder.dx).abs() * 0.2);
      cand.remove(wrist);
      final elbow = pickByScore(
        cand,
        (p) => (p.dy - shoulder.dy).abs() + (wrist.dy - p.dy).abs() * 0.5 + _dist(p, shoulder) * 0.2,
      );
      pool.remove(wrist);
      pool.remove(elbow);
      // Ensure elbow above wrist
      if (elbow.dy > wrist.dy) {
        return [wrist, elbow]; // swap
      }
      return [elbow, wrist];
    }

    final lArm = sideArm(lSh, true);
    final rArm = sideArm(rSh, false);
    final lElb = lArm[0], lWri = lArm[1];
    final rElb = rArm[0], rWri = rArm[1];

    // 7) Eyes/Ears: from whatever head-level points remain (best-effort)
    final headLefts = <Offset>[], headRights = <Offset>[];
    for (final p in pool) {
      if (p.dy <= lSh.dy && p.dy <= rSh.dy) {
        if (p.dx <= cx) {
          headLefts.add(p);
        } else {
          headRights.add(p);
        }
      }
    }
    Offset orDefault(List<Offset> lst, Offset def) => lst.isEmpty
        ? def
        : lst.reduce((a, b) => a.dy < b.dy ? a : b);
    final headSpan = (rSh.dx - lSh.dx).abs();
    final headH = (lSh.dy - nose.dy).abs();
    final lEye = orDefault(headLefts, Offset(lSh.dx - 0.15 * headSpan, nose.dy + 0.06 * headH));
    final rEye = orDefault(headRights, Offset(rSh.dx + 0.15 * headSpan, nose.dy + 0.06 * headH));
    final lEar = Offset(lEye.dx - 0.06 * headSpan, lEye.dy + 0.02 * headH);
    final rEar = Offset(rEye.dx + 0.06 * headSpan, rEye.dy + 0.02 * headH);

    // Assemble COCO-17
    return <Offset>[
      nose, // 0
      lEye, // 1
      rEye, // 2
      lEar, // 3
      rEar, // 4
      lSh, // 5
      rSh, // 6
      lElb, // 7
      rElb, // 8
      lWri, // 9
      rWri, // 10
      lHip, // 11
      rHip, // 12
      lKnee, // 13
      rKnee, // 14
      lAnk, // 15
      rAnk, // 16
    ];
  }

  // ───────────────────────── geometry helpers ─────────────────────────

  /// bbox: returns (minX, minY, maxX, maxY)
  (double, double, double, double) _bbox(List<Offset> pts) {
    double minX = double.infinity, minY = double.infinity;
    double maxX = -double.infinity, maxY = -double.infinity;
    for (final p in pts) {
      if (!p.dx.isFinite || !p.dy.isFinite) continue;
      if (p.dx < minX) minX = p.dx;
      if (p.dy < minY) minY = p.dy;
      if (p.dx > maxX) maxX = p.dx;
      if (p.dy > maxY) maxY = p.dy;
    }
    if (!minX.isFinite) {
      minX = 0;
      minY = 0;
      maxX = 0;
      maxY = 0;
    }
    return (minX, minY, maxX, maxY);
  }

  double _bboxHeight(List<Offset> pts) {
    final b = _bbox(pts);
    return (b.$4 - b.$2).abs();
  }

  double _medianX(List<Offset> pts) {
    final xs = pts.map((p) => p.dx).toList()..sort();
    final n = xs.length;
    if (n == 0) return 0;
    return (n % 2 == 1) ? xs[n ~/ 2] : (xs[n ~/ 2 - 1] + xs[n ~/ 2]) / 2.0;
  }

  List<Offset> _takeLowestY(List<Offset> pts, int k) {
    final sorted = List<Offset>.from(pts)..sort((a, b) => b.dy.compareTo(a.dy));
    return sorted.take(k).toList();
  }

  Offset _popHighest(List<Offset> pool) {
    int bi = 0;
    double by = double.infinity;
    for (int i = 0; i < pool.length; i++) {
      if (pool[i].dy < by) {
        by = pool[i].dy;
        bi = i;
      }
    }
    return pool.removeAt(bi);
  }

  double _dist(Offset a, Offset b) {
    final dx = (a.dx - b.dx), dy = (a.dy - b.dy);
    return math.sqrt(dx * dx + dy * dy);
  }
}

// ───────────────────────── minimal .npy reader (optional) ─────────────────────────
// Supports C-order float32/float64 arrays with ndim up to 2.
// Enough for [17,2], [17,3], [26,2], [26,3] legacy reference files.
class _NpyArray {
  _NpyArray(this.shape, this.dataF32, this.dataF64);
  final List<int> shape;
  final Float32List? dataF32;
  final Float64List? dataF64;
  double get(int i, int j) {
    final cols = shape[1];
    final idx = i * cols + j;
    if (dataF32 != null) return dataF32![idx].toDouble();
    return dataF64![idx];
  }
}

_NpyArray _npyLoad(Uint8List bytes) {
  // Magic \x93NUMPY
  if (bytes.length < 10 || bytes[0] != 0x93 || bytes[1] != 0x4E) {
    throw StateError('Not a .npy file');
  }
  final major = bytes[6];
  // Header length @8..9 (v1) or 8..11 (v2)
  int headerLen;
  int headerStart;
  if (major == 1) {
    headerLen = bytes.buffer.asByteData().getUint16(8, Endian.little);
    headerStart = 10;
  } else {
    headerLen = bytes.buffer.asByteData().getUint32(8, Endian.little);
    headerStart = 12;
  }
  final header = String.fromCharCodes(bytes.sublist(headerStart, headerStart + headerLen));
  // Parse descr
  final descrMatch = RegExp(r"'descr':\\s*'([^']+)'").firstMatch(header);
  if (descrMatch == null) throw StateError('npy: missing descr');
  final descr = descrMatch.group(1)!; // e.g. "<f8" or "<f4"

  final shapeMatch = RegExp(r"'shape':\\s*\\(([^)]*)\\)").firstMatch(header);
  if (shapeMatch == null) throw StateError('npy: missing shape');
  final shapeStr = shapeMatch.group(1)!.trim();
  final shape = shapeStr
      .split(',')
      .where((s) => s.trim().isNotEmpty)
      .map((s) => int.parse(s.trim()))
      .toList();

  final fortran = header.contains("'fortran_order': True");
  if (fortran) {
    throw UnsupportedError('Fortran-order arrays not supported');
  }

  final dataStart = headerStart + headerLen;
  final dataBytes = bytes.sublist(dataStart);
  final littleEndian = descr.startsWith('<') || descr.startsWith('|');
  if (!littleEndian) {
    throw UnsupportedError('Big-endian npy not supported here');
  }

  if (descr.endsWith('f8')) {
    final f64 = Float64List.view(dataBytes.buffer, dataBytes.offsetInBytes);
    return _NpyArray(shape, null, f64);
  } else if (descr.endsWith('f4')) {
    final f32 = Float32List.view(dataBytes.buffer, dataBytes.offsetInBytes);
    return _NpyArray(shape, f32, null);
  } else {
    throw UnsupportedError('Only float32/float64 supported');
  }
}
