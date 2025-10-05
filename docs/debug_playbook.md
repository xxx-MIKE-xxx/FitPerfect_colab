# FitPerfect Debug Playbook

This playbook helps diagnose issues across the FitPerfect capture pipeline quickly. Enable the one-time debug switch to collect structured logs and artifacts, then follow the stage-by-stage guidance below.

## 0. Enable debug mode

Introduce a single boolean `debug` flag (e.g., surfaced in app settings). When enabled, each stage:

- Emits structured JSON logs to both the console and a session-specific log file.
- Saves small inspection artifacts (first frames, tiny JSON snippets, PNG/JPG overlays) into the session directory.

**Session folder**: `Documents/FitPerfect/<sessionId>/`, obtained via `getApplicationDocumentsDirectory()` on iOS/Android. Treat this as the canonical location for all per-run artifacts.

---

## 1. Camera capture & YUV → RGB

**Log every ~30 frames, plus first and last frames:**

- `CAM`: camera resolution, rotation/orientation, frame index `t`, timestamp (ms).
- `YUV2RGB`: converter path (native vs Dart), average conversion time, output tensor shape `[H, W, 3]`.

**Persist (first frame):** `frame_0000_rgb.jpg` for visual sanity (orientation & color order).

**Failure signatures:**

| Symptom | Likely cause |
| --- | --- |
| Skin appears blue/green | Incorrect YUV→RGB conversion order (channel swap). |
| Severe frame drops | Conversion executing on UI thread/Dart—prefer native or FFI implementation. |

---

## 2. Letterbox (YOLO input 640×640, pad=114)

**Log first 5 frames, then every ~60:**

- `LBX`: source height/width, scale `r`, padding offsets `dw`, `dh`, fill color `[114,114,114]`, output shape `640×640`.

**Persist (first frame):** `t0000_letterbox.jpg`.

> YOLOv8 expects neutral gray padding (114). Keep this constant—do not auto-tune.

**Failure signatures:**

- Incorrect `r/dw/dh` results in wrong un-padding later (boxes cling to edges).
- Missing padding distorts subjects and harms detections.

---

## 3. YOLO inference & decode

**Log every frame:**

- `YOLO_IN`: input tensor shape (target `[1,3,640,640]`), normalization (RGB/255 confirmation).
- `YOLO_EP`: active execution provider (CoreML/XNNPACK/CPU) and per-frame latency.
- `YOLO_RAW`: output tensor shape (e.g., `[1,84,8400]`) and first three values.
- `YOLO_DET`: candidate count pre-NMS, top score, selected bbox in letterbox coordinates, normalization flag.

**Persist:** `yolo_decode.txt` with entries such as `t=0 valid=3087 chosen_score=0.71 bbox=[x1,y1,x2,y2]`.

**Notes:**

- iOS logs like `IsInputSupported: Dynamic shape is not supported …` indicate CoreML fallback to CPU. Mitigate by enabling the CoreML flag `onlyAllowStaticInputShapes` or running YOLO/RTM exclusively on XNNPACK.

**Failure signatures:**

| Symptom | Likely cause |
| --- | --- |
| Person detected late (~50/60 frames) | Threshold/NMS too strict or faulty un-letterboxing. |
| BBox `[0,…,0,…]` | Normalized outputs never rescaled/un-padded. |

---

## 4. Un-letterbox to original coordinates

For each selected detection compute:

```
x1_img = (x1_lb - dw) / r
x2_img = (x2_lb - dw) / r
y1_img = (y1_lb - dh) / r
y2_img = (y2_lb - dh) / r
```

**Log:** final image-space bbox per frame.

**Persist:** overlay every ~60 frames (`bbox_overlay_t0060.jpg`) showing the bbox on the original RGB frame.

**Failure signatures:** incorrect `r/dw/dh` yields misaligned crops, confusing downstream models.

---

## 5. Crop → 256×192 & RTMPose (SimCC)

**Log every frame or every 5 frames:**

- `CROP`: selected bbox, adjusted crop rectangle, scale factor (e.g., ×1.25), output size `256×192`.
- `RTM_IN`: tensor shape `[1,3,256,192]`, preprocessing (RGB/255).
- `RTM_EP`: execution provider, timings.
- `RTM_OUT`: SimCC tensor shapes (e.g., `X=384`, `Y=512`), top 3 peak confidences.

**Persist:**

- `t0000_cropped.jpg` (first frame).
- `t00xx_kpts_on_crop.jpg` (first frame with valid keypoints, overlayed).

**Notes:**

- RTMPose uses SimCC decoding; ensure decode logic matches tensor sizes and split ratio (commonly 2.0).

**Failure signatures:**

| Symptom | Likely cause |
| --- | --- |
| Near-zero confidences | Wrong color order/normalization or SimCC decode bug. |
| Skeleton shifted/scaled | Incorrect mapping from crop to image coordinates. |

---

## 6. Map keypoints back to image & overlay

**Log every ~10 frames:**

- `KPTS_IMG`: min/max `(x,y)` across 17 joints, mean confidence, `x_ptp`/`y_ptp` ranges.
- Record tracker status (`TRACKER`) when fallback boxes are reused.

**Persist:** `overlay_t00xx.jpg` combining original image, bbox, and skeleton.

**Expected ranges:**

- `conf_mean` ≫ 0.002.
- `x_ptp`/`y_ptp` should span hundreds of pixels on 1080p frames.

---

## 7. Save 2D sequence (JSONL)

**Writer flow:**

1. `2D_SAVE_OPEN`: log path to `coco_2d.jsonl` (once per session).
2. `2D_SAVE_WRITE`: per frame JSON line: 
   ```json
   {
     "t": 0.033,
     "img_w": 1080,
     "img_h": 1920,
     "bbox_norm": [x1/W, y1/H, x2/W, y2/H],
     "kpt_h36m": [[x,y,conf], ... 17],
     "yolo_units": "normalized",
     "lbx": {"r":0.58,"dw":12,"dh":48}
   }
   ```
3. `2D_SAVE_FLUSH` & `2D_SAVE_CLOSE`: confirm bytes written after recording stops.

**Reader flow:** search `Documents/FitPerfect/<sessionId>/coco_2d.jsonl`. If legacy paths (`…/poses/<sessionId>/2d/frames.jsonl`) remain, log which was used.

**Failure signatures:**

- Missing summary data: mismatched session IDs, incorrect directory, or writer never flushed/closed.

---

## 8. MotionBERT (post-record 3D)

**Pre-run logs:**

- `MB_PREP`: total frames, keypoint mapping (COCO→H36M if needed), normalization strategy, window size (243) and stride.
- `MB_EP`: provider choice and timing.

**Run logs:**

- `MB_RUN`: input shape `[1,243,17,3]`, elapsed ms.
- `MB_OUT`: Z-range (min/max), representative frame sample.

**Persist:** `out_3d.json`, optional `mb_input_seq.npy`, and `mb_quick_stats.json` (frame count, z-range).

> MotionBERT expects H36M-17 keypoints. Match Python normalization: center & scale by min-side, maintain consistent padding/truncation for the 243-frame window.

---

## 9. Execution providers & model shapes (iOS)

**Session start logs:**

- `ORT_ENV`: ONNX Runtime version & build flags.
- `EP_CHAIN`: ordered provider chain (e.g., `CoreML(staticOnly=true) → XNNPACK → CPU`).
- `COREML`: include `onlyAllowStaticInputShapes` and `createMLProgram` flags.

**Handling warnings:**

- Route dynamic-shape nodes to XNNPACK/CPU or enable static-only CoreML to avoid spam like `Dynamic shape not supported`.
- Optionally, pre-fix dynamic shapes in ONNX via official ORT tooling.

---

## 10. End-to-end integrity self-check

Provide a debug-only "Session Self-Check" button that reports:

1. **Paths & IDs:** session folder used by writer/reader, recursive file listing.
2. **2D sanity:** parse `coco_2d.jsonl`, compute frame count, `conf_mean`, `x_ptp`, `y_ptp`.
3. **3D sanity:** if `out_3d.json` exists, print frame count and `z_min/z_max`.
4. **Overlay presence:** confirm `overlay_t00xx.jpg`; if missing, suggest enabling overlay exports.

---

## 11. Suggested log keys & examples

Recommended JSON log keys:

```
CAM, YUV2RGB, LBX, YOLO_IN, YOLO_EP, YOLO_RAW, YOLO_DET, UNPAD,
CROP, RTM_IN, RTM_EP, RTM_OUT, SIMCC, KPTS_IMG, OVERLAY,
2D_SAVE_OPEN, 2D_SAVE_WRITE, 2D_SAVE_CLOSE,
MB_PREP, MB_EP, MB_RUN, MB_OUT,
EP_CHAIN, COREML_WARN, SESSION_PATHS
```

**Sample entries:**

```
YOLO_DET {"t":14,"valid":3087,"maxScore":0.711,"bbox_lb":[0,0.86,0,2.98],"norm":true}
UNPAD    {"t":14,"bbox_img":[150,260,960,1780],"r":0.582,"dw":12,"dh":48}
KPTS_IMG {"t":14,"conf_mean":0.74,"x_ptp":690,"y_ptp":1520}
2D_SAVE_WRITE {"t":14,"path":".../FitPerfect/2025-10.../coco_2d.jsonl","bytes":312}
MB_OUT   {"frames":243,"joints":17,"z_min":-0.42,"z_max":0.61}
```

---

## 12. Common failure → fix table

| Symptom | Likely cause | Where to inspect | Suggested fix |
| --- | --- | --- | --- |
| Person appears only after 50–60 frames | YOLO threshold too high, aggressive NMS, or incorrect un-letterboxing | `YOLO_DET`, `UNPAD` | Reduce confidence/NMS thresholds; verify `r/dw/dh`; ensure normalized outputs mapped back to pixels. |
| Keypoints cluster near `(0,0)` or confidences ≈ 0 | Broken crop or SimCC decode | `CROP`, `RTM_IN`, `RTM_OUT` | Correct crop rectangle; match SimCC tensor sizes (384/512) & split ratio; check color order and `/255` normalization. |
| iOS logs "Dynamic shape is not supported" | CoreML execution provider declining dynamic nodes | `EP_CHAIN`, device console | Enable CoreML static-only flag; route dynamic ops to XNNPACK; or fix ONNX shapes offline. |
| `coco_2d.jsonl` missing on summary | Session path mismatch or writer never flushed | `2D_SAVE_OPEN/CLOSE`, `SESSION_PATHS` | Ensure consistent `<sessionId>`; search both live and legacy directories; flush and close file sink. |
| MotionBERT Z range extreme | Incorrect MotionBERT normalization | `MB_PREP` | Follow H36M-17 mapping and normalization (center + min-side scale). |

---

## 13. Optional: performance sampling

- Record per-stage timings: `YUV2RGB_ms`, `YOLO_ms`, `RTM_ms`, `MB_ms`.
- Compare cumulative latency against frame budget (e.g., 33 ms @ 30 FPS) and log overruns.
- Attribute workloads to execution providers (per model or per node, when available). XNNPACK is a strong fallback accelerator on iOS/Android when CoreML/NNAPI reject certain ops.

