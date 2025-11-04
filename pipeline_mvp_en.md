# Pipeline MVP: YOLO Live + Offline RTMPose and MotionBERT

A smallest-cost, minimal-viable implementation that runs YOLO live during recording, saves video and detections, and then performs RTMPose and MotionBERT offline. Paths, file names, and core record formats remain compatible with the original spec.

---

## Scope

Included
- Live YOLO person detection during recording
- Saving MP4 video and YOLO detections to disk
- Offline 2D: RTMPose on sampled frames using saved YOLO boxes
- Offline 3D: MotionBERT on COCO→H36M converted sequences
- Same session-folder layout and file names

Excluded
- Preflight benchmark and mode decision
- Telemetry, dynamic sampling, backlog handling
- Optional debug artifacts not listed below

---

## Session folder layout

iOS
- <App Documents>/FitPerfect/<sessionId>/
- Example: /var/mobile/Containers/Data/Application/<UUID>/Documents/FitPerfect/<sessionId>/

Android
- <App files>/FitPerfect/<sessionId>/
- Example: /data/user/0/<pkg>/files/FitPerfect/<sessionId>/ (base from getApplicationDocumentsDirectory())

Files inside the session folder
- video.mp4
- yolo_person.jsonl
- coco_2d.jsonl
- out_3d.json
- meta.json

---

## Constants

Sampling interval used offline
- Δt = 125 ms (target ~8 FPS)

YOLO model input
- 640 × 640 with letterbox

RTMPose model input
- 256 × 192 with letterbox

MotionBERT windowing
- T = 243, stride = 81

---

## Live step: record and detect (YOLO)

Goal
- While recording MP4, run YOLO on the latest camera frame only (buffer size 1). For every processed frame, select the largest person bbox and write a JSONL record.

Live processing
- Perform letterbox to 640 × 640 for YOLO input
- Run YOLO and decode detections
- Pick the largest bbox of class person
- Convert bbox back to raw-frame pixels (invert letterbox) and also save as normalized coordinates
- Append one line to yolo_person.jsonl with timing and bbox

Record format for yolo_person.jsonl
- One line equals one processed frame

```json
{
  "t": 3.200,
  "ts_ms": 3200,
  "frame_idx": 96,
  "frame_size": [720, 1280],
  "bbox_raw": [x1, y1, x2, y2],
  "bbox_norm": [x1_W, y1_H, x2_W, y2_H],
  "score": 0.89
}
```

Notes
- Only the largest person is tracked
- If no person is detected on a frame, nothing is written for that frame

---

## Offline step 1: build 2D keypoints (RTMPose)

Input
- video.mp4
- yolo_person.jsonl

Sampling
- Fixed step by wall-clock timestamp using Δt = 125 ms
- For each sample time, find the nearest or last prior YOLO record by ts_ms
- If the last good bbox is older than 500 ms, mark the sample as missing and skip RTMPose for it

Per-sample processing
- Crop the raw frame to the selected bbox
- Letterbox crop to 256 × 192 and store letterbox parameters r, dw, dh, in_w, in_h, out
- Run RTMPose on the letterboxed crop
- Un-letterbox keypoints back to raw-frame pixels and store [x_px, y_px, conf]

Output file coco_2d.jsonl
- One line equals one processed sample

```json
{
  "t": 3.200,
  "ts_ms": 3200,
  "frame_idx": 96,
  "frame_size": [720, 1280],
  "bbox_norm": [0.22, 0.12, 0.55, 0.88],
  "lb_params": {"r": 0.56, "dw": 40, "dh": 12, "in_w": 720, "in_h": 1280, "out": [256, 192]},
  "kpt_coco": [[x_px, y_px, conf], "... 17 entries ..."]
}
```

---

## Offline step 2: compute 3D (MotionBERT)

Preparation
- Resample coco_2d.jsonl to the fixed Δt timeline if needed
- Convert COCO-17 to H36M-17 joint layout
- Normalize by centering at pelvis and scaling by min(W, H) / 2
- Pack temporal windows: T = 243, stride = 81

Inference
- Run MotionBERT on each window and merge outputs

Output file out_3d.json
- Final 3D sequence with shape [T_total, 17, 3] in H36M layout, pelvis-centered, normalized to approximately [-1, 1]

---

## COCO-17 to H36M-17 mapping

COCO indices
- 0 nose, 1 leye, 2 reye, 3 lear, 4 rear, 5 lsho, 6 rsho, 7 lelb, 8 relb, 9 lwri, 10 rwri, 11 lhip, 12 rhip, 13 lknee, 14 rknee, 15 lank, 16 rank

H36M order required by MotionBERT
- Pelvis, RHip, RKnee, RAnkle, LHip, LKnee, LAnkle, Spine1, Neck, Head, Site, LShoulder, LElbow, LWrist, RShoulder, RElbow, RWrist

Mapping in raw-frame pixels
- Pelvis = midpoint of LHip(11) and RHip(12)
- RHip(12) → RHip
- RKnee(14) → RKnee
- RAnkle(16) → RAnkle
- LHip(11) → LHip
- LKnee(13) → LKnee
- LAnkle(15) → LAnkle
- Spine1 = midpoint of Pelvis and Neck
- Neck = midpoint of LShoulder(5) and RShoulder(6)
- Head = average of leye(1), reye(2), lear(3), rear(4); if missing, average eyes only
- Site = Nose(0)
- LShoulder(5) → LShoulder
- LElbow(7) → LElbow
- LWrist(9) → LWrist
- RShoulder(6) → RShoulder
- RElbow(8) → RElbow
- RWrist(10) → RWrist

---

## Minimal pseudocode

Live recorder and YOLO

```dart
// initialize video recording to <session>/video.mp4
// initialize yolo session

while (recording) {
  final frame = camera.latestFrame();      // buffer size 1
  if (frame == null) continue;

  final det = yolo.run(letterbox(frame));  // 640x640
  final main = pickLargestPerson(det);
  if (main != null) {
    final bboxRaw = invertLetterbox(main.bbox, frame.size);
    writeJsonl("yolo_person.jsonl", {
      "t": (nowMs() - startMs)/1000.0,
      "ts_ms": nowMs(),
      "frame_idx": frame.index,
      "frame_size": [frame.w, frame.h],
      "bbox_raw": bboxRaw,
      "bbox_norm": normalize(bboxRaw, frame.size),
      "score": main.score
    });
  }
}
```

Offline RTMPose and MotionBERT

```dart
final samples = timelineFrom(startMs, endMs, dtMs: 125);
for (final s in samples) {
  final frame = frameAt(video, s.ts_ms);
  final yoloRec = nearestYoloBefore("yolo_person.jsonl", s.ts_ms, maxAgeMs: 500);
  if (yoloRec == null) continue;

  final crop = cropFrame(frame, yoloRec.bbox_raw);
  final lb = letterboxParams(crop.size, outW: 256, outH: 192);
  final inp = applyLetterbox(crop, lb);
  final kps = rtmPose.run(inp);
  final kpsRaw = unletterboxKeypoints(kps, lb);

  appendJsonl("coco_2d.jsonl", {
    "t": s.t,
    "ts_ms": s.ts_ms,
    "frame_idx": frame.index,
    "frame_size": [frame.w, frame.h],
    "bbox_norm": yoloRec.bbox_norm,
    "lb_params": lb.toJson(),
    "kpt_coco": kpsRaw
  });
}

// then: resample → coco→h36m → normalize → window (T=243, stride=81) → MotionBERT → write out_3d.json
```

---

## meta.json example

```json
{
  "session_id": "2025-10-09T12-00-01Z_x1y2z3",
  "device": {"platform": "iOS", "ep": "coreml", "ram_gb": 6},
  "video": {"fps": 30, "size": [720, 1280], "duration_s": 12.4},
  "sampling": {"dt_ms": 125, "strategy": "timestamp"},
  "models": {
    "yolo": {"name": "yolov8n-person.onnx", "input": [640, 640]},
    "rtmpose": {"name": "rtmpose-s.onnx", "input": [256, 192]},
    "motionbert": {"name": "MB_ft_h36m.bin", "clip_len_max": 243}
  },
  "counts": {"frames_2d": 98, "clips_3d": 3}
}
```

---

## Dependencies

- ONNX Runtime Mobile (Core ML on iOS, NNAPI on Android; CPU fallback)
- Flutter plugin: flutter_onnxruntime or onnxruntime
- Models: yolov8n-person.onnx, rtmpose-s.onnx, MB_ft_h36m.bin

---

## Deliverables for MVP

- video.mp4 and yolo_person.jsonl recorded live in the session folder
- coco_2d.jsonl produced offline from video and YOLO records
- out_3d.json produced offline by MotionBERT
- meta.json with basic run info
