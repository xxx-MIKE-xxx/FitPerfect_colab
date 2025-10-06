// lib/shared/services/motionbert_runner.dart
//
// MotionBertRunner: runs the post-record 3D stage exactly like the Python pipeline.
// Input  : Documents/FitPerfect/<sessionId>/coco_2d.jsonl   (written by LivePoseEngine)
// Output : Documents/FitPerfect/<sessionId>/out_3d.json     (and optional NPY debug files)
//
// Steps (mirrors run_rgb_to_motionbert3d_patched_v5.py):
//  1) Read COCO-17 [x,y,conf] keypoints per frame from jsonl
//  2) Convert COCO-17 → H36M-17 (same ordering as Python)
//  3) Pad/trim to T=243 (pad last frame if short; take last 243 if long)
//  4) Normalize XY to [-1,1] using (cx,cy)=(W/2,H/2) and scale=min(W,H)/2
//  5) Run MotionBERT ONNX: input [1,243,17,3] float32, output [1,243,17,3]
//  6) If rootRelative=true, zero pelvis (joint 0) in 3D output
//  7) Save out_3d.json (compatible with check_and_visualize_3d.py)
//
// Note: We treat ONNX Runtime session as `dynamic` via your ort_session.dart wrapper.
// If the input name isn't 'input', adjust the `_runMb(...)` helper below.
//
// Copyright (c) FitPerfect

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' show Size;

import 'package:path_provider/path_provider.dart';

import 'ort_session.dart'; // OrtManager.fromAsset(...)
import 'pipeline_debug_recorder.dart';

class MotionBertRunner {
  MotionBertRunner({
    this.modelAssetPath = 'assets/models/motionbert_3d_243.onnx',
    this.debug = false,
  });

  /// Path to MotionBERT ONNX/ORT model inside assets.
  final String modelAssetPath;
  final bool debug;

  /// Run MotionBERT 3D for a finished live session.
  /// [frameSize] must match the original image size used during 2D (width x height).
  /// Returns the created out_3d.json file.
  Future<File> run({
    required String sessionId,
    required Size frameSize,
    bool rootRelative = true,
    bool writeNpy = false,
  }) async {
    // Resolve session directory (Documents first, then Temp fallback).
    final Directory docs = await getApplicationDocumentsDirectory();
    final Directory temp = await getTemporaryDirectory();
    final dirDocs = Directory('${docs.path}/FitPerfect/$sessionId');
    final dirTemp = Directory('${temp.path}/FitPerfect/$sessionId');
    final Directory sessionDir =
        await dirDocs.exists() ? dirDocs : (await dirTemp.exists() ? dirTemp : dirDocs);
    if (!(await sessionDir.exists())) {
      await sessionDir.create(recursive: true);
    }

    final PipelineDebugRecorder? dbg = debug
        ? PipelineDebugRecorder(
            enabled: true,
            sessionId: sessionId,
            sessionDir: sessionDir,
          )
        : null;

    dbg?.log('SESSION_PATHS', {
      'sessionId': sessionId,
      'sessionDir': sessionDir.path,
      'model': modelAssetPath,
    });

    
    // Robust 2D lookup: try primary then legacy path
    final File primary2D = File('${sessionDir.path}/coco_2d.jsonl');
    final File legacy2D  = File('${docs.path}/FitPerfect/poses/$sessionId/2d/frames.jsonl');
    dbg?.log('MB_2D_LOOKUP', {
      'primary': primary2D.path,
      'legacy' : legacy2D.path,
    });
    final File file2D = await primary2D.exists()
        ? primary2D
        : (await legacy2D.exists() ? legacy2D : primary2D);
    if (!await file2D.exists()) {
      // Log both attempted paths before throwing
      final msg = 'Missing 2D keypoints: tried primary=${primary2D.path} and legacy=${legacy2D.path}';
      dbg?.log('MB_2D_LOOKUP_FAIL', {'message': msg});
      throw StateError(msg);
    }

    if (!await file2D.exists()) {
      throw StateError('Missing coco_2d.jsonl at ${file2D.path}');
    }

    try {
      // --- 1) Read COCO-17 sequence from jsonl
      final seqCoco = <List<List<double>>>[]; // [T][17][3]
      await for (final line in file2D.openRead()
          .transform(utf8.decoder)
          .transform(const LineSplitter())) {
        if (line.trim().isEmpty) continue;
        final obj = json.decode(line) as Map<String, dynamic>;
        final kpts = (obj['kpt_coco'] as List)
            .map<List<double>>((e) => (e as List).map((v) => (v as num).toDouble()).toList())
            .toList();
        if (kpts.length == 17) {
          seqCoco.add(kpts);
        }
      }
      if (seqCoco.isEmpty) {
        throw StateError('No frames found in coco_2d.jsonl');
      }

      final int W = frameSize.width.round();
      final int H = frameSize.height.round();

      // --- 2) COCO-17 → H36M-17 (xy + conf per joint)
      final seqH36 = <List<List<double>>>[]; // [T][17][3]
      for (final kpts in seqCoco) {
        final h36 = _cocoToH36M(kpts);
        seqH36.add(h36);
      }

      // --- 3) Pad/trim to 243
      const int T = 243;
      List<List<List<double>>> seq = List.of(seqH36);
      if (seq.length < T) {
        final tail =
            seq.isNotEmpty ? seq.last : List.generate(17, (_) => [0.0, 0.0, 0.0]);
        while (seq.length < T) {
          seq.add(_deepCopyFrame(tail));
        }
      } else if (seq.length > T) {
        // take last 243 frames
        seq = seq.sublist(seq.length - T);
      }

      // --- 4) Normalize XY to [-1,1], keep conf
      final Float32List mbIn = Float32List(T * 17 * 3);
      final double s = (math.min(W, H) / 2.0);
      final double cx = W / 2.0;
      final double cy = H / 2.0;
      dbg?.log('MB_PREP', {
        'sessionId': sessionId,
        'frames': seqCoco.length,
        'window': T,
        'stride': 81,
        'normalize': {
          'center': [cx, cy],
          'scale': s,
        },
      });
      int idx = 0;
      for (int t = 0; t < T; t++) {
        final fr = seq[t];
        for (int j = 0; j < 17; j++) {
          final x = (fr[j][0] - cx) / s;
          final y = (fr[j][1] - cy) / s;
          final c = fr[j][2];
          mbIn[idx++] = x.toDouble();
          mbIn[idx++] = y.toDouble();
          mbIn[idx++] = c.toDouble();
        }
      }

      // Optional: write mb_input_seq.npy (debug parity)
      if (writeNpy) {
        await _writeNpy('${sessionDir.path}/mb_input_seq.npy', mbIn, [T, 17, 3]);
      }

      final loadTimer = Stopwatch()..start();
      final dynamic mb = await OrtManager.fromAsset(modelAssetPath);
      final double loadMs = loadTimer.elapsedMicroseconds / 1000.0;
      dbg?.log('MB_EP', {
        'sessionId': sessionId,
        'model': modelAssetPath,
        'providers': ['coreml', 'nnapi', 'xnnpack', 'cpu'],
        'loadMs': double.parse(loadMs.toStringAsFixed(3)),
      });

      // --- 5) Run MotionBERT ONNX
      final Float32List mbInBatched = Float32List(1 * T * 17 * 3)..setAll(0, mbIn);
      final runTimer = Stopwatch()..start();
      final dynamic out = await _runMb(mb, mbInBatched);
      final double runMs = runTimer.elapsedMicroseconds / 1000.0;
      dbg?.log('MB_RUN', {
        'sessionId': sessionId,
        'elapsedMs': double.parse(runMs.toStringAsFixed(3)),
        'inputShape': [1, T, 17, 3],
      });
      final Float32List flat = _asFloat32(out);

      // Expect [1,T,17,3] → flatten length = 1*243*17*3
      if (flat.length != 1 * T * 17 * 3) {
        throw StateError('Unexpected MotionBERT output length: ${flat.length}');
      }

      // Parse to [T][17][3]
      final List<List<List<double>>> X3D = List.generate(
        T, (_) => List.generate(17, (_) => List.filled(3, 0.0)),
      );
      double zMin = double.infinity;
      double zMax = -double.infinity;
      int p = 0;
      for (int t = 0; t < T; t++) {
        for (int j = 0; j < 17; j++) {
          final x = flat[p++].toDouble();
          final y = flat[p++].toDouble();
          final z = flat[p++].toDouble();
          X3D[t][j][0] = x;
          X3D[t][j][1] = y;
          X3D[t][j][2] = z;
          zMin = math.min(zMin, z);
          zMax = math.max(zMax, z);
        }
      }

      // --- 6) Root-relative zeroing (pelvis joint 0)
      if (rootRelative) {
        for (int t = 0; t < T; t++) {
          X3D[t][0][0] = 0.0;
          X3D[t][0][1] = 0.0;
          X3D[t][0][2] = 0.0;
        }
      }

      dbg?.log('MB_OUT', {
        'sessionId': sessionId,
        'frames': T,
        'z_min': double.parse(zMin.toStringAsFixed(4)),
        'z_max': double.parse(zMax.toStringAsFixed(4)),
      });

      if (dbg != null && dbg.enabled) {
        final stats = {
          'frames_2d': seqCoco.length,
          'window': T,
          'stride': 81,
          'z_min': zMin,
          'z_max': zMax,
        };
        await dbg.saveText(
          'mb_quick_stats.json',
          const JsonEncoder.withIndent('  ').convert(stats),
        );
      }

      // Optional: write mb_output_3d.npy
      if (writeNpy) {
        final Float32List outFlat = Float32List(T * 17 * 3);
        int q = 0;
        for (int t = 0; t < T; t++) {
          for (int j = 0; j < 17; j++) {
            outFlat[q++] = X3D[t][j][0].toDouble();
            outFlat[q++] = X3D[t][j][1].toDouble();
            outFlat[q++] = X3D[t][j][2].toDouble();
          }
        }
        await _writeNpy('${sessionDir.path}/mb_output_3d.npy', outFlat, [T, 17, 3]);
      }

      // --- 7) Save out_3d.json (compatible with your visualizer)
      final outJson = {
        "T": T,
        "h36m_order": const [
          "Pelvis","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle",
          "Spine1","Neck","Head","Site","LShoulder","LElbow","LWrist",
          "RShoulder","RElbow","RWrist"
        ],
        "coords_3d": X3D,
      };
      final outFile = File('${sessionDir.path}/out_3d.json');
      await outFile.writeAsString(const JsonEncoder.withIndent('  ').convert(outJson));
      dbg?.log('MB_OUT_FILE', {
        'sessionId': sessionId,
        'path': outFile.path,
      });
      return outFile;
    } finally {
      await dbg?.close();
    }
  }

  // ---------- helpers ----------

  // Try to run session with common input name(s).
  Future<dynamic> _runMb(dynamic session, Float32List input) async {
    // common name
    try {
      return await session.run({"input": input});
    } catch (_) {
      // some wrappers accept first (and only) key as arbitrary
      try {
        return await session.run({ "0": input });
      } catch (_) {
        // last resort: pass list
        return await session.run([input]);
      }
    }
  }

  // Convert various tensor wrappers to Float32List.
  Float32List _asFloat32(dynamic v) {
    if (v is Float32List) return v;
    if (v is List) {
      // Many plugins return List<OrtValue> or List<List<float>>; try flatten
      if (v.isEmpty) return Float32List(0);
      final first = v.first;
      if (first is Float32List) {
        // Assume single-output model
        return first;
      }
      if (first is List) {
        // flatten recursively
        final buf = <double>[];
        void walk(dynamic x) {
          if (x is List) {
            for (final e in x) walk(e);
          } else if (x is num) {
            buf.add(x.toDouble());
          } else if (x is Float32List) {
            buf.addAll(x);
          } else if (x is Int32List) {
            buf.addAll(x.map((e) => e.toDouble()));
          }
        }
        walk(v);
        return Float32List.fromList(buf);
      }
      // List<OrtValue> wrapper with .data
      try {
        final firstData = (first as dynamic).data;
        if (firstData is Float32List) return firstData;
        if (firstData is List) {
          return Float32List.fromList(firstData.map((e) => (e as num).toDouble()).toList());
        }
      } catch (_) {}
    }
    // Map form: get first value
    if (v is Map) {
      if (v.isNotEmpty) return _asFloat32(v.values.first);
    }
    // Generic wrapper with .data or .value
    try {
      final d = (v as dynamic).data;
      if (d is Float32List) return d;
      if (d is List) return Float32List.fromList(d.map((e) => (e as num).toDouble()).toList());
    } catch (_) {}
    try {
      final d = (v as dynamic).value;
      if (d is Float32List) return d;
      if (d is List) return Float32List.fromList(d.map((e) => (e as num).toDouble()).toList());
    } catch (_) {}
    throw StateError('Unsupported MotionBERT output type: ${v.runtimeType}');
  }

  // COCO-17 → H36M-17 with confidence mapping.
  // COCO indices: 0 nose,1 leye,2 reye,3 lear,4 rear,5 lsho,6 rsho,7 lelb,8 relb,9 lwri,10 rwri,11 lhip,12 rhip,13 lknee,14 rknee,15 lank,16 rank
  // H36M order we use: Pelvis,RHip,RKnee,RAnkle,LHip,LKnee,LAnkle,Spine1,Neck,Head,Site,LShoulder,LElbow,LWrist,RShoulder,RElbow,RWrist
  List<List<double>> _cocoToH36M(List<List<double>> coco) {
    double px(int i) => coco[i][0];
    double py(int i) => coco[i][1];
    double pc(int i) => coco[i][2];

    final lhip = [px(11), py(11)];
    final rhip = [px(12), py(12)];
    final pelvis = [(lhip[0] + rhip[0]) / 2.0, (lhip[1] + rhip[1]) / 2.0];

    final lsho = [px(5), py(5)];
    final rsho = [px(6), py(6)];
    final neck = [(lsho[0] + rsho[0]) / 2.0, (lsho[1] + rsho[1]) / 2.0];
    final spine1 = [(pelvis[0] + neck[0]) / 2.0, (pelvis[1] + neck[1]) / 2.0];

    final leye = [px(1), py(1)];
    final reye = [px(2), py(2)];
    final nose = [px(0), py(0)];
    final head = (pc(1) > 0 && pc(2) > 0)
        ? [(leye[0] + reye[0]) / 2.0, (leye[1] + reye[1]) / 2.0]
        : [nose[0], nose[1]];
    final site = [nose[0], nose[1]];

    List<List<double>> h = List.generate(17, (_) => [0.0, 0.0, 0.0]);
    // coords (xy) + conf
    void setJ(int j, double x, double y, double c) { h[j][0] = x; h[j][1] = y; h[j][2] = c; }

    final cPelvis = (pc(11) + pc(12)) / 2.0;
    final cNeck   = (pc(5) + pc(6)) / 2.0;
    final cSpine1 = (cPelvis + cNeck) / 2.0;
    final cHead   = (pc(1) > 0 && pc(2) > 0) ? (pc(1) + pc(2)) / 2.0 : pc(0);
    final cSite   = pc(0);

    setJ(0,  pelvis[0], pelvis[1], cPelvis);
    setJ(1,  px(12), py(12), pc(12)); // RHip
    setJ(2,  px(14), py(14), pc(14)); // RKnee
    setJ(3,  px(16), py(16), pc(16)); // RAnkle
    setJ(4,  px(11), py(11), pc(11)); // LHip
    setJ(5,  px(13), py(13), pc(13)); // LKnee
    setJ(6,  px(15), py(15), pc(15)); // LAnkle
    setJ(7,  spine1[0], spine1[1], cSpine1); // Spine1
    setJ(8,  neck[0], neck[1], cNeck);       // Neck
    setJ(9,  head[0], head[1], cHead);       // Head
    setJ(10, site[0], site[1], cSite);       // Site
    setJ(11, px(5), py(5), pc(5));  // LShoulder
    setJ(12, px(7), py(7), pc(7));  // LElbow
    setJ(13, px(9), py(9), pc(9));  // LWrist
    setJ(14, px(6), py(6), pc(6));  // RShoulder
    setJ(15, px(8), py(8), pc(8));  // RElbow
    setJ(16, px(10),py(10),pc(10)); // RWrist
    return h;
  }

  List<List<double>> _deepCopyFrame(List<List<double>> fr) =>
      fr.map((j) => [j[0].toDouble(), j[1].toDouble(), j[2].toDouble()]).toList();

  // Minimal NPY writer for float32, little-endian, C-order.
  Future<void> _writeNpy(String path, Float32List data, List<int> shape) async {
    final file = File(path);
    final sink = file.openWrite();
    // magic string
    sink.add([0x93] + 'NUMPY'.codeUnits);
    // version 1.0
    sink.add([1, 0]);
    final descr = "{'descr': '<f4', 'fortran_order': False, 'shape': (${shape.join(', ')},), }";
    // pad header to 16-byte alignment
    final hdrBytes = utf8.encode(descr);
    final headerLen = hdrBytes.length + 1; // newline
    final padLen = ((16 - ((10 + headerLen) % 16)) % 16);
    final totalLen = headerLen + padLen;
    // 2-byte little-endian header length
    final hl = totalLen;
    sink.add([hl & 0xFF, (hl >> 8) & 0xFF]);
    sink.add(hdrBytes);
    sink.add(List.filled(padLen + 1, 0x20)); // spaces incl. newline at end
    // data
    sink.add(data.buffer.asUint8List());
    await sink.flush();
    await sink.close();
  }
}