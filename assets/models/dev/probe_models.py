#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Probe YOLOv8 (detection) and RTMPose (top-down 2D pose) ONNX models to discover:
- I/O names, dtypes, shapes
- YOLOv8: input size, raw output layout (e.g., [1,84,N] or [1,N,84]),
          number of classes, presence of NonMaxSuppression op (NMS in-graph)
- RTMPose: input size (expect 3x256x192), output type (SimCC vs heatmap),
           keypoint count K, SimCC split ratio inferred from output sizes

Also includes a COCO-17 -> H36M-17 mapping helper for MotionBERT.

Usage:
  python probe_models.py --yolo yolov8n.onnx --rtmpose rtmpose-m_256x192.onnx \
      --save report.json

Requires: onnx, onnxruntime, numpy
"""

import argparse, json, sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import onnx
import onnxruntime as ort

# ---------- Utilities ----------

def ort_io(sess: ort.InferenceSession) -> Dict[str, Any]:
    def io_to_dict(io):
        return {"name": io.name, "dtype": io.type, "shape": [int(d) if isinstance(d, int) else None for d in io.shape]}
    return {
        "inputs": [io_to_dict(i) for i in sess.get_inputs()],
        "outputs": [io_to_dict(o) for o in sess.get_outputs()]
    }

def run_dummy(sess: ort.InferenceSession, input_shape: List[int]) -> List[np.ndarray]:
    # Build a small dummy tensor matching the first input shape.
    shape = [1 if d in (None, "None") else int(d) for d in input_shape]
    x = np.zeros(shape, dtype=np.float32)
    out = sess.run(None, {sess.get_inputs()[0].name: x})
    return [np.asarray(o) for o in out]

def has_nms_node(onnx_model: onnx.ModelProto) -> bool:
    return any(n.op_type == "NonMaxSuppression" for n in onnx_model.graph.node)

# ---------- YOLOv8 inspector ----------

def inspect_yolo(onnx_path: Path) -> Dict[str, Any]:
    model = onnx.load(str(onnx_path))
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    io = ort_io(sess)

    # Infer input H,W (NCHW expected)
    in0 = sess.get_inputs()[0]
    ishape = [int(d) if isinstance(d, int) else None for d in in0.shape]  # [B,C,H,W]
    _, c, h, w = ishape
    # Dummy inference to get concrete output array shapes
    outs = run_dummy(sess, ishape)
    oshapes = [list(o.shape) for o in outs]

    # Try to parse raw detection head
    # Typical detection export: a single tensor [1, 84, N] or [1, N, 84] (84 = 4 + 80 COCO classes)
    num_classes = None
    raw_layout = None
    anchors = None
    for s in oshapes:
        if len(s) == 3:
            b, a, b2 = s  # could be [1,84,N] or [1,N,84]
            if a in (84, 85) or b2 in (84, 85):  # 4+nc (nc=80) or sometimes 85 (with objectness)
                if a in (84, 85):
                    raw_layout = "[B, 84/85, N]"
                    num_classes = a - 4 if a in (84, 85) else None
                    anchors = b2
                else:
                    raw_layout = "[B, N, 84/85]"
                    num_classes = b2 - 4 if b2 in (84, 85) else None
                    anchors = a

    info = {
        "path": str(onnx_path.resolve()),
        "framework": "YOLOv8 (detection)",
        "ort_io": io,
        "input_nchw": [1, c, h, w],
        "output_shapes": oshapes,
        "raw_head_layout": raw_layout,           # None if atypical export
        "num_classes_inferred": num_classes,     # e.g., 80 for COCO
        "num_candidates_inferred": anchors,      # e.g., 8400 for 640x640
        "nms_in_graph": has_nms_node(model),     # True if export includes NMS op
    }
    return info

# ---------- RTMPose inspector ----------

def inspect_rtmpose(onnx_path: Path) -> Dict[str, Any]:
    model = onnx.load(str(onnx_path))
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    io = ort_io(sess)
    in0 = sess.get_inputs()[0]
    ishape = [int(d) if isinstance(d, int) else None for d in in0.shape]  # [B,C,H,W] expected
    _, c, h, w = ishape

    outs = run_dummy(sess, ishape)
    oshapes = [list(o.shape) for o in outs]

    # Determine SimCC vs heatmap:
    # SimCC: TWO outputs of shape [B, K, Sx] and [B, K, Sy] (names often 'simcc_x', 'simcc_y')
    # Heatmap: ONE output of shape [B, K, H', W']
    model_type = None
    keypoints = None
    simcc_ratio_x = None
    simcc_ratio_y = None

    if len(oshapes) == 2 and all(len(s) == 3 for s in oshapes):
        # SimCC
        model_type = "SimCC"
        b1, k1, sx = oshapes[0]
        b2, k2, sy = oshapes[1]
        keypoints = k1 if k1 == k2 else None
        if w and h and sx and sy:
            simcc_ratio_x = float(sx) / float(w)
            simcc_ratio_y = float(sy) / float(h)
    elif len(oshapes) == 1 and len(oshapes[0]) == 4:
        # Heatmap
        model_type = "Heatmap"
        _, keypoints, hh, ww = oshapes[0]

    info = {
        "path": str(onnx_path.resolve()),
        "framework": "RTMPose (top-down 2D pose)",
        "ort_io": io,
        "input_nchw": [1, c, h, w],
        "output_shapes": oshapes,
        "head_type": model_type,                 # 'SimCC' or 'Heatmap'
        "num_keypoints_inferred": keypoints,     # e.g., 17 for COCO body
        "simcc_split_ratio": [simcc_ratio_x, simcc_ratio_y] if model_type == "SimCC" else None
    }
    return info

# ---------- COCO-17 -> H36M-17 mapping (for MotionBERT) ----------

# COCO 17 order:
# 0 Nose, 1 LEye, 2 REye, 3 LEar, 4 REar,
# 5 LShoulder, 6 RShoulder, 7 LElbow, 8 RElbow, 9 LWrist, 10 RWrist,
# 11 LHip, 12 RHip, 13 LKnee, 14 RKnee, 15 LAnkle, 16 RAnkle
#
# H36M 17 order (official):
# 0 Pelvis, 1 RHip, 2 RKnee, 3 RAnkle, 4 LHip, 5 LKnee, 6 LAnkle,
# 7 Spine1, 8 Neck, 9 Head, 10 Site, 11 LShoulder, 12 LElbow, 13 LWrist,
# 14 RShoulder, 15 RElbow, 16 RWrist
#
# Notes:
# - H36M lacks an explicit "top of head" in COCO, so we approximate "Head" and "Site".
# - We'll set H36M[9] (Head) = COCO Nose; H36M[10] (Site) = COCO Nose as a safe fallback.
#   (If you have a detector with explicit "Head top" (e.g., Halpe-26), use that instead.)
#
def coco17_to_h36m17(kpts_coco_xyc: np.ndarray) -> np.ndarray:
    """
    kpts_coco_xyc: [17, 3] as (x, y, conf) in COCO order
    returns: [17, 3] H36M order
    """
    out = np.zeros((17, 3), dtype=np.float32)
    # Helper mids
    lhip = kpts_coco_xyc[11, :2]; rhip = kpts_coco_xyc[12, :2]
    lsho = kpts_coco_xyc[5,  :2]; rsho = kpts_coco_xyc[6,  :2]
    pelvis = (lhip + rhip) / 2.0
    neck   = (lsho + rsho) / 2.0
    spine1 = (pelvis + neck) / 2.0
    nose   = kpts_coco_xyc[0, :2]

    # Confidences: take min/mean of sources where we average
    conf_pelvis = min(kpts_coco_xyc[11,2], kpts_coco_xyc[12,2])
    conf_neck   = min(kpts_coco_xyc[5,2],  kpts_coco_xyc[6,2])
    conf_spine1 = min(conf_pelvis, conf_neck)

    # Fill H36M
    out[0,:2], out[0,2] = pelvis, conf_pelvis         # Pelvis
    out[1,:2], out[1,2] = kpts_coco_xyc[12,:2], kpts_coco_xyc[12,2]  # RHip
    out[2,:2], out[2,2] = kpts_coco_xyc[14,:2], kpts_coco_xyc[14,2]  # RKnee
    out[3,:2], out[3,2] = kpts_coco_xyc[16,:2], kpts_coco_xyc[16,2]  # RAnkle
    out[4,:2], out[4,2] = kpts_coco_xyc[11,:2], kpts_coco_xyc[11,2]  # LHip
    out[5,:2], out[5,2] = kpts_coco_xyc[13,:2], kpts_coco_xyc[13,2]  # LKnee
    out[6,:2], out[6,2] = kpts_coco_xyc[15,:2], kpts_coco_xyc[15,2]  # LAnkle
    out[7,:2], out[7,2] = spine1, conf_spine1                           # Spine1
    out[8,:2], out[8,2] = neck, conf_neck                               # Neck
    out[9,:2], out[9,2] = nose, kpts_coco_xyc[0,2]                      # Head (approx)
    out[10,:2], out[10,2] = nose, kpts_coco_xyc[0,2]                    # Site (approx)
    out[11,:2], out[11,2] = kpts_coco_xyc[5,:2],  kpts_coco_xyc[5,2]    # LShoulder
    out[12,:2], out[12,2] = kpts_coco_xyc[7,:2],  kpts_coco_xyc[7,2]    # LElbow
    out[13,:2], out[13,2] = kpts_coco_xyc[9,:2],  kpts_coco_xyc[9,2]    # LWrist
    out[14,:2], out[14,2] = kpts_coco_xyc[6,:2],  kpts_coco_xyc[6,2]    # RShoulder
    out[15,:2], out[15,2] = kpts_coco_xyc[8,:2],  kpts_coco_xyc[8,2]    # RElbow
    out[16,:2], out[16,2] = kpts_coco_xyc[10,:2], kpts_coco_xyc[10,2]   # RWrist
    return out

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", type=str, required=True, help="Path to YOLOv8 ONNX (e.g., yolov8n.onnx)")
    ap.add_argument("--rtmpose", type=str, required=True, help="Path to RTMPose ONNX (e.g., rtmpose-m_256x192.onnx)")
    ap.add_argument("--save", type=str, default="report.json", help="Where to save the discovery report")
    args = ap.parse_args()

    yolo_info = inspect_yolo(Path(args.yolo))
    rtm_info  = inspect_rtmpose(Path(args.rtmpose))

    report = {
        "yolov8": yolo_info,
        "rtmpose": rtm_info,
        "notes": {
            "motionbert_input": "[B, T, 17, C] with H36M-17 joints; convert from COCO-17 before feeding",
            "h36m_order": ["Pelvis","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle","Spine1","Neck","Head","Site","LShoulder","LElbow","LWrist","RShoulder","RElbow","RWrist"]
        }
    }
    Path(args.save).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()

