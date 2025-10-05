#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run MotionBERT 3D ONNX (243 frames) and auto-confirm model I/O + rootrel.

Usage
-----
1) Place the ONNX file locally, e.g. motionbert_3d_243.onnx
2) (Optional) Use the bundled sample JSON (H36M-17 layout), or point to your own.
3) Run:
   python run_motionbert_243.py --model /path/to/motionbert_3d_243.onnx \
       --json /path/to/sample_keypoints_h36m17.json

   You can also omit --json to auto-generate synthetic keypoints.

This script will:
- Load the ONNX model with onnxruntime
- Print input/output NAMES, SHAPES, and DTYPES
- Try C=3 (x,y,conf) then C=2 (x,y) if needed
- Run one forward pass on a 243-frame clip
- Decide if outputs are **root-relative** by checking pelvis ~ [0,0,0] across frames
- Save a small JSON report next to the input JSON / script folder

JSON schema (expected)
----------------------
{
  "layout": "h36m17",
  "width": 256,                # optional if normalized inputs only
  "height": 256,               # optional if normalized inputs only
  "keypoints": [               # length T
    [                          # frame 0
      {"x": 0.1, "y": -0.2, "conf": 0.9},  # 17 entries, H36M order
      ... (17 joints total)
    ],
    ... (T frames)
  ]
}

Assumptions
-----------
- Input layout is Human3.6M 17-joint order (pelvis index 0).
- Coordinates are normalized to ~[-1, 1] by default. Use --pixel if your JSON
  is in pixel coordinates; the script will normalize them.
- ONNX model expects shape [B, T, 17, C] with B=1, T<=243, C in {2,3}.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np

try:
    import onnxruntime as ort
except Exception as e:
    print("ERROR: onnxruntime is required. Install with: pip install onnxruntime", file=sys.stderr)
    raise

H36M_JOINTS = [
    "Pelvis","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle",
    "Spine1","Neck","Head","Site","LShoulder","LElbow","LWrist",
    "RShoulder","RElbow","RWrist"
]

def generate_synthetic_h36m(T: int = 243, normalized: bool = True,
                            width: int = 256, height: int = 256,
                            seed: int = 42) -> Dict[str, Any]:
    """Create a toy sequence of H36M-17 keypoints."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2*np.pi, T, endpoint=False)
    # Base pose roughly centered
    base_xy = rng.normal(loc=0.0, scale=0.05, size=(17, 2))
    conf = np.clip(rng.normal(loc=0.95, scale=0.03, size=(T, 17, 1)), 0.5, 1.0)
    # Simple periodic motion added
    motion = np.stack([0.1*np.sin(t), 0.1*np.cos(t)], axis=-1)  # [T,2]
    xy = base_xy[None, :, :] + motion[:, None, :] * (rng.uniform(0.5, 1.5, size=(17,))[None, :, None])
    # Pelvis at center-ish
    xy[:, 0, :] *= 0.2
    # Normalize or convert to pixels
    if normalized:
        xyn = np.clip(xy, -1.0, 1.0)
        coords = np.concatenate([xyn, conf], axis=-1)  # [T,17,3]
        width_out, height_out = None, None
    else:
        # map [-1,1] -> pixels around center
        xyp = (xy * (min(width, height) / 2.0)) + np.array([width/2.0, height/2.0])
        coords = np.concatenate([xyp, conf], axis=-1)
        width_out, height_out = int(width), int(height)

    frames = []
    for ti in range(T):
        joints = []
        for j in range(17):
            x, y, c = coords[ti, j, 0], coords[ti, j, 1], coords[ti, j, 2]
            joints.append({"x": float(x), "y": float(y), "conf": float(c)})
        frames.append(joints)

    return {
        "layout": "h36m17",
        "width": width_out,
        "height": height_out,
        "keypoints": frames
    }

def load_keypoints_json(path: Path) -> Tuple[np.ndarray, Optional[int], Optional[int]]:
    data = json.loads(Path(path).read_text())
    assert data.get("layout", "").lower() in {"h36m17", "h36m"}, "JSON must be H36M-17 order"
    kp = data["keypoints"]
    T = len(kp)
    assert T > 0, "No frames"
    assert len(kp[0]) == 17, "Expected 17 joints (H36M)"
    arr = np.zeros((T, 17, 3), dtype=np.float32)
    for ti in range(T):
        for j in range(17):
            joint = kp[ti][j]
            arr[ti, j, 0] = float(joint.get("x", 0.0))
            arr[ti, j, 1] = float(joint.get("y", 0.0))
            arr[ti, j, 2] = float(joint.get("conf", 1.0))
    width = data.get("width")
    height = data.get("height")
    if width is not None: width = int(width)
    if height is not None: height = int(height)
    return arr, width, height

def maybe_normalize_xy(xy_conf: np.ndarray, pixel: bool, width: Optional[int], height: Optional[int]) -> np.ndarray:
    """xy_conf: [T,17,3] with (x,y,conf). If pixel==True, normalize to [-1,1]."""
    if not pixel:
        return xy_conf
    assert width and height, "Pixel mode requires width/height in JSON"
    # inverse of infer_wild's pixel->normalized conversion:
    # normalized = (pixels - (w/2,h/2)) / (min(w,h)/2)
    out = xy_conf.copy()
    out[:, :, 0] = (xy_conf[:, :, 0] - (width / 2.0)) / (min(width, height) / 2.0)
    out[:, :, 1] = (xy_conf[:, :, 1] - (height / 2.0)) / (min(width, height) / 2.0)
    return out

def to_batch_input(xy_conf: np.ndarray, channels: int) -> np.ndarray:
    T = xy_conf.shape[0]
    if channels == 2:
        x = xy_conf[:, :, :2]
    else:
        x = xy_conf  # 3
    x = x.astype(np.float32)[None, ...]  # [1,T,17,C]
    return x

def ort_io_summary(sess: "ort.InferenceSession") -> dict:
    def io_to_dict(io):
        return {
            "name": io.name,
            "shape": [int(d) if isinstance(d, int) else (None if d is None else str(d)) for d in io.shape],
            "dtype": io.type
        }
    inputs = [io_to_dict(i) for i in sess.get_inputs()]
    outputs = [io_to_dict(o) for o in sess.get_outputs()]
    return {"inputs": inputs, "outputs": outputs}

def try_infer(sess, x_btc, input_name: str) -> np.ndarray:
    y = sess.run(None, {input_name: x_btc})[0]  # assume first output is 3D joints
    return y

def detect_root_relative(y_btj3: np.ndarray, atol: float = 1e-3) -> dict:
    """y: [1,T,17,3] or [T,17,3]. Returns stats + boolean."""
    y = y_btj3
    if y.ndim == 3:
        y = y[None, ...]
    pelvis = y[0, :, 0, :]  # H36M pelvis index 0
    norms = np.linalg.norm(pelvis, axis=-1)  # [T]
    is_rootrel = bool(np.allclose(pelvis, 0.0, atol=atol))
    stats = {
        "pelvis_mean_norm": float(np.mean(norms)),
        "pelvis_max_norm": float(np.max(norms)),
        "pelvis_std_norm": float(np.std(norms)),
        "frames": int(y.shape[1])
    }
    return {"is_root_relative": is_rootrel, "stats": stats}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='Path to motionbert_3d_243.onnx')
    ap.add_argument('--json', default=None, help='Path to keypoints JSON (H36M-17). If omitted, synthetic data is generated.')
    ap.add_argument('--pixel', action='store_true', help='If your JSON x/y are pixels, add this to normalize to [-1,1].')
    ap.add_argument('--t', type=int, default=243, help='Clip length to use (<=243).')
    ap.add_argument('--channels', choices=['auto','2','3'], default='auto', help='Use 3 (x,y,conf), 2 (x,y), or auto-try 3 then 2.')
    ap.add_argument('--report', default='mb243_report.json', help='Where to save the summary report.')
    args = ap.parse_args()

    T = min(args.t, 243)
    if args.json is None:
        data = generate_synthetic_h36m(T=T, normalized=True)
    else:
        # Load user JSON
        arr, w, h = load_keypoints_json(Path(args.json))
        arr = arr[:T]  # clip
        data = {"layout":"h36m17","width":w,"height":h,"keypoints": []}
        for ti in range(arr.shape[0]):
            frame = []
            for j in range(17):
                frame.append({"x": float(arr[ti,j,0]), "y": float(arr[ti,j,1]), "conf": float(arr[ti,j,2])})
            data["keypoints"].append(frame)

    # Build input tensor
    xy_conf = np.array([[(kp["x"], kp["y"], kp.get("conf",1.0)) for kp in fr] for fr in data["keypoints"]], dtype=np.float32)
    xy_conf = maybe_normalize_xy(xy_conf, pixel=args.pixel, width=data.get("width"), height=data.get("height"))

    # Init ORT
    sess = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    io_meta = ort_io_summary(sess)
    print("\n== ONNX I/O ==") 
    print(json.dumps(io_meta, indent=2))

    # Decide input name (first input) and channels
    in_name = io_meta['inputs'][0]['name'] if io_meta['inputs'] else sess.get_inputs()[0].name

    def run_with_channels(C: int):
        x = to_batch_input(xy_conf, channels=C)
        y = try_infer(sess, x, in_name)
        return x, y

    used_channels = None
    y = None
    if args.channels in ('3','auto'):
        try:
            x, y = run_with_channels(3)
            used_channels = 3
        except Exception as e:
            if args.channels == '3':
                raise
            print("[Auto] C=3 failed, trying C=2...", file=sys.stderr)
    if y is None:
        x, y = run_with_channels(2)
        used_channels = 2

    # Sanity on output shape
    y = np.array(y)
    if y.ndim == 3:
        # [T,17,3] -> add batch
        y = y[None, ...]
    assert y.shape[-2] == 17 and y.shape[-1] == 3, f"Unexpected output shape {y.shape}, expected [...,17,3]"

    # Detect root-relative
    rootrel_info = detect_root_relative(y)
    print("\n== Root-relative check ==")
    print(json.dumps(rootrel_info, indent=2))

    # Save report
    rpt = {
        "model_path": str(Path(args.model).resolve()),
        "input_output": io_meta,
        "clip_length": int(y.shape[1]),
        "used_channels": used_channels,
        "rootrel": rootrel_info,
    }
    Path(args.report).write_text(json.dumps(rpt, indent=2))
    print(f"\nSaved report to {args.report}")

if __name__ == '__main__':
    main()
