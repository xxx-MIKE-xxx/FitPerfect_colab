#!/usr/bin/env python3
# probe_yolo_rtm_diag.py
# Diagnose YOLOv8 (ONNX) -> RTMPose (ONNX) stages independently.
# - Probes YOLO decode variants and dumps overlays to see which one yields valid boxes.
# - Optionally probes RTMPose on detected crops with multiple preprocess flavors.
#
# Usage examples:
#   python probe_yolo_rtm_diag.py --video dev/video.mp4 --yolo yolov8n.onnx --out_dir debug_probe --probe yolo
#   python probe_yolo_rtm_diag.py --video dev/video.mp4 --yolo yolov8n.onnx --rtmpose rtmpose-m_256x192.onnx --out_dir debug_probe --probe rtm --max_frames 16
#
# Requires: onnxruntime, opencv-python, numpy
import argparse, os, json, math, sys
from pathlib import Path
import numpy as np
import cv2
import onnxruntime as ort

def letterbox(img, new_shape=(640, 640), color=(114,114,114)):
    shape = img.shape[:2]  # (h,w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (w,h)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def xywh2xyxy(x):
    y = np.empty_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def draw_box(im, box, color=(0,255,0), label=None):
    x1,y1,x2,y2 = [int(round(v)) for v in box]
    cv2.rectangle(im, (x1,y1), (x2,y2), color, 2)
    if label:
        cv2.putText(im, label, (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

def read_video_frames(path, max_frames=None):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, im = cap.read()
        if not ret:
            break
        frames.append(im)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames

def postproc_variants(letter_w=640, letter_h=640):
    # Four decode guesses for bbox coordinates
    # A: first 4 are xywh (pixels in 640 letterbox); B: first 4 are xyxy (pixels);
    # C/D: same but assume they are normalized to [0,1] -> multiply by 640.
    return {
        "xywh_pixels": lambda b: xywh2xyxy(b),
        "xyxy_pixels": lambda b: b,
        "xywh_norm":   lambda b: xywh2xyxy(b * letter_w),
        "xyxy_norm":   lambda b: b * letter_w,
    }

def unletterbox_xyxy(boxes, r, dw, dh, img_w, img_h):
    boxes = boxes.copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / r
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / r
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img_w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img_h - 1)
    return boxes

def probe_yolo(video, yolo_path, out_dir, max_frames=32, conf_thres=0.05):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    frames = read_video_frames(video, max_frames=max_frames)
    if not frames: raise SystemExit(f"Could not read frames from {video}")
    sess = ort.InferenceSession(yolo_path, providers=['CPUExecutionProvider'])
    in_name = sess.get_inputs()[0].name

    stats = {k: {"frames": 0, "has_person": 0, "valid_boxes": 0} for k in postproc_variants().keys()}
    for ti, im in enumerate(frames):
        # Preprocess
        im_letter, r, (dw, dh) = letterbox(im, (640, 640))
        im_rgb = cv2.cvtColor(im_letter, cv2.COLOR_BGR2RGB)
        x = (im_rgb.astype(np.float32) / 255.0).transpose(2,0,1)[None,...]
        out = sess.run(None, {in_name: x})[0]  # [1,84,8400]
        pred = out[0].T  # [N,84]
        raw_boxes = pred[:, :4]
        cls_logits = pred[:, 4:]
        cls_probs = sigmoid(cls_logits)
        cls_ids = np.argmax(cls_probs, axis=1)
        cls_scores = cls_probs[np.arange(cls_probs.shape[0]), cls_ids]

        variants = postproc_variants()
        for name, fn in variants.items():
            # decode and map back
            boxes_xyxy = fn(raw_boxes)
            boxes_xyxy = unletterbox_xyxy(boxes_xyxy, r, dw, dh, im.shape[1], im.shape[0])
            # person filter
            m = (cls_ids == 0) & (cls_scores >= conf_thres)
            boxes = boxes_xyxy[m]
            scores = cls_scores[m]
            stats[name]["frames"] += 1
            if len(boxes) > 0:
                # positive area
                wh = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
                nvalid = int(np.sum(wh > 4.0))  # > 2x2 px
                stats[name]["valid_boxes"] += nvalid
                stats[name]["has_person"] += int(nvalid > 0)
                # dump one overlay per variant for first few frames
                if ti < 6:
                    ov = im.copy()
                    idx = np.argmax(scores)
                    draw_box(ov, boxes[idx], (0,255,0), f"{name}:{scores[idx]:.2f}")
                    cv2.imwrite(str(out_dir / f"yolo_{name}_t{ti:03d}.jpg"), ov)

    with open(out_dir / "yolo_probe_summary.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("YOLO probe summary saved to", out_dir / "yolo_probe_summary.json")

def softmax(a, axis=-1):
    a = a - np.max(a, axis=axis, keepdims=True)
    ea = np.exp(a)
    return ea / np.sum(ea, axis=axis, keepdims=True)

def simcc_decode(simcc_x, simcc_y, split_ratio=2.0):
    px = softmax(simcc_x, axis=2)
    py = softmax(simcc_y, axis=2)
    x_idx = np.argmax(px, axis=2)
    y_idx = np.argmax(py, axis=2)
    x = x_idx.astype(np.float32) / float(split_ratio)
    y = y_idx.astype(np.float32) / float(split_ratio)
    conf = np.sqrt(np.max(px, axis=2) * np.max(py, axis=2)).astype(np.float32)
    coords = np.stack([x, y], axis=-1)
    return coords, conf

def crop_to_aspect(img, box_xyxy, out_hw=(256, 192), scale=1.25):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0
    bw = (x2 - x1) * scale; bh = (y2 - y1) * scale
    out_h, out_w = out_hw; target_ar = out_w / out_h
    if bw / bh > target_ar: bh = bw / target_ar
    else: bw = bh * target_ar
    x1n = max(0, int(round(cx - bw / 2))); y1n = max(0, int(round(cy - bh / 2)))
    x2n = min(w - 1, int(round(cx + bw / 2))); y2n = min(h - 1, int(round(cy + bh / 2)))
    crop = img[y1n:y2n, x1n:x2n]
    if crop.size == 0: return None, None
    resized = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    rect = (x1n, y1n, x2n - x1n, y2n - y1n)
    return resized, rect

def probe_rtm(video, yolo_path, rtm_path, out_dir, max_frames=16, conf_thres=0.05):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    frames = read_video_frames(video, max_frames=max_frames)
    if not frames: raise SystemExit(f"Could not read frames from {video}")
    # Simple person bbox using the most permissive YOLO variant that gives area
    sess = ort.InferenceSession(yolo_path, providers=['CPUExecutionProvider'])
    in_name = sess.get_inputs()[0].name
    rtm = ort.InferenceSession(rtm_path, providers=['CPUExecutionProvider'])
    rtm_in = rtm.get_inputs()[0].name

    # preprocess flavors for RTMPose
    flavors = {
        "rgb_ms": dict(ms=True, rgb=True),
        "bgr_ms": dict(ms=True, rgb=False),
        "rgb_255": dict(ms=False, rgb=True),
        "bgr_255": dict(ms=False, rgb=False),
    }
    results = {k: {"frames": 0, "mean_conf": 0.0} for k in flavors}

    for ti, im in enumerate(frames):
        im_letter, r, (dw, dh) = letterbox(im, (640, 640))
        im_rgb = cv2.cvtColor(im_letter, cv2.COLOR_BGR2RGB)
        x = (im_rgb.astype(np.float32) / 255.0).transpose(2,0,1)[None,...]
        out = sess.run(None, {in_name: x})[0]
        pred = out[0].T
        raw_boxes = pred[:, :4]; cls_logits = pred[:, 4:]
        cls_probs = sigmoid(cls_logits)
        cls_ids = np.argmax(cls_probs, axis=1)
        cls_scores = cls_probs[np.arange(cls_probs.shape[0]), cls_ids]

        best_box = None; best_score = -1
        for fn in postproc_variants().values():
            boxes_xyxy = unletterbox_xyxy(fn(raw_boxes), r, dw, dh, im.shape[1], im.shape[0])
            m = (cls_ids == 0) & (cls_scores >= conf_thres)
            boxes = boxes_xyxy[m]; scores = cls_scores[m]
            if len(boxes) == 0: continue
            # choose largest area
            areas = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
            idx = int(np.argmax(areas))
            if scores[idx] > best_score and areas[idx] > 100.0:
                best_box, best_score = boxes[idx], float(scores[idx])

        if best_box is None:
            continue

        crop, rect = crop_to_aspect(im, best_box, out_hw=(256, 192), scale=1.25)
        if crop is None: continue

        # run RTMPose with 4 preprocess flavors and save minivis
        H,W = 256,192
        for name, cfg in flavors.items():
            im_in = crop.copy()
            if cfg["rgb"]: im_in = cv2.cvtColor(im_in, cv2.COLOR_BGR2RGB)
            im_in = im_in.astype(np.float32)
            if cfg["ms"]:
                mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
                std  = np.array([58.395, 57.12, 57.375], dtype=np.float32)
                im_in = (im_in - mean) / std
            else:
                im_in = im_in / 255.0
            x_in = im_in.transpose(2,0,1)[None,...]
            simcc_x, simcc_y = rtm.run(None, {rtm_in: x_in})
            coords, conf = simcc_decode(simcc_x, simcc_y, split_ratio=2.0)
            results[name]["frames"] += 1
            results[name]["mean_conf"] += float(conf.mean())
            # tiny overlay
            ov = crop.copy()
            for p in coords[0]:
                cv2.circle(ov, (int(round(p[0])), int(round(p[1]))), 2, (0,255,0), -1, cv2.LINE_AA)
            cv2.imwrite(str(out_dir / f"rtm_{name}_t{ti:03d}.jpg"), ov)

    # normalize mean_conf by frames
    for k,v in results.items():
        if v["frames"] > 0:
            v["mean_conf"] = v["mean_conf"] / v["frames"]
    with open(out_dir / "rtm_probe_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("RTMPose probe summary saved to", out_dir / "rtm_probe_summary.json")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True, type=str)
    ap.add_argument('--yolo', required=True, type=str)
    ap.add_argument('--rtmpose', type=str, default=None)
    ap.add_argument('--out_dir', type=str, default='debug_probe')
    ap.add_argument('--probe', type=str, choices=['yolo','rtm'], default='yolo')
    ap.add_argument('--max_frames', type=int, default=32)
    ap.add_argument('--conf', type=float, default=0.05)
    args = ap.parse_args()

    if args.probe == 'yolo':
        probe_yolo(args.video, args.yolo, args.out_dir, max_frames=args.max_frames, conf_thres=args.conf)
    else:
        if not args.rtmpose:
            raise SystemExit("--rtmpose is required when --probe rtm")
        probe_rtm(args.video, args.yolo, args.rtmpose, args.out_dir, max_frames=args.max_frames, conf_thres=args.conf)

if __name__ == "__main__":
    main()
