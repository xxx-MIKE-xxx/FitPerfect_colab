#!/usr/bin/env python3
"""
Discover MotionBERT 3D ONNX I/O and behavioral conventions.

Usage:
  pip install onnx onnxruntime numpy
  python discover_motionbert.py --model motionbert3d.onnx --skeleton skeletons.json --skeleton-name auto --report report.json

What it does:
  1) Prints exact I/O names/shapes/dtypes (authoritative from ONNX).
  2) Scans graph constants & ops for mean/std, /255, ×1000 (mm), etc.
  3) Runs synthetic probes to test:
       - pelvis-centering (translation invariance),
       - scale normalization (scale invariance),
       - output units (meters vs millimeters) via magnitude analysis.
  4) Writes a machine-readable JSON report with evidence and confidence.

Notes:
  - The input here is (1, T, 17, 3) (from your log). We assume the last
    channel is (x, y, score). Score is set to 1.0 by default.
  - Synthetic probes are heuristic but reproducible and quantized into
    clear pass/fail with confidence scores and numeric evidence.
"""

import argparse, json, math, sys, collections
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper, mapping

# ---------- Utilities ----------

def close(a, b, rtol=1e-5, atol=1e-8):
    return np.isclose(a, b, rtol=rtol, atol=atol)

def nearly(value, targets, rtol=1e-4, atol=1e-8):
    for t in targets:
        if close(float(value), float(t), rtol=rtol, atol=atol):
            return True
    return False

def safe_float(x):
    try:
        return float(np.asarray(x).reshape(()))
    except Exception:
        return None

def stable_percentile(arr, q):
    arr = np.asarray(arr).astype(np.float64).reshape(-1)
    if arr.size == 0:
        return np.nan
    return float(np.percentile(arr, q))


@dataclass
class IOInfo:
    name: str
    np_dtype: str
    onnx_dtype: int
    shape: List

# ---------- Skeleton handling ----------

@dataclass
class Skeleton:
    name: str
    joints: List[str]
    edges: List[Tuple[int, int]]
    root_index: int

    @staticmethod
    def from_dict(d: Dict):
        return Skeleton(
            name=d["name"],
            joints=d["joints"],
            edges=[tuple(e) for e in d["edges"]],
            root_index=int(d["root_index"]),
        )

def load_skeletons(path: str) -> Dict[str, Skeleton]:
    with open(path, "r") as f:
        meta = json.load(f)
    out = {}
    for d in meta.get("skeletons", []):
        s = Skeleton.from_dict(d)
        out[s.name] = s
    return out

def pick_skeleton(skeletons: Dict[str, Skeleton], requested: str, num_joints: int) -> Skeleton:
    if requested != "auto":
        sk = skeletons.get(requested)
        if sk is None:
            raise ValueError(f"Requested skeleton '{requested}' not found in JSON.")
        if len(sk.joints) != num_joints:
            print(f"[WARN] Skeleton '{sk.name}' joint count {len(sk.joints)} != model input count {num_joints}. Proceeding anyway.")
        return sk
    # Auto: find any skeleton with matching joint count; prefer common ones.
    ranked = sorted(skeletons.values(), key=lambda s: (abs(len(s.joints) - num_joints), 0 if s.name in ("h36m17","coco17") else 1))
    if not ranked:
        raise ValueError("No skeletons found in the JSON.")
    # If first not exact match, warn.
    if len(ranked[0].joints) != num_joints:
        print(f"[WARN] No skeleton with {num_joints} joints. Using '{ranked[0].name}' with {len(ranked[0].joints)}.")
    return ranked[0]

# ---------- ONNX Introspection ----------

def get_io_info(model_path: str) -> Tuple[List[IOInfo], List[IOInfo], onnx.ModelProto]:
    m = onnx.load(model_path)
    graph = m.graph
    def to_ioinfo(vi):
        t = vi.type.tensor_type
        onnx_dtype = t.elem_type
        np_dtype = mapping.TENSOR_TYPE_TO_NP_TYPE.get(onnx_dtype, None)
        dims = [(d.dim_param if d.dim_param else d.dim_value) for d in t.shape.dim]
        return IOInfo(vi.name, str(np_dtype), onnx_dtype, dims)
    ins = [to_ioinfo(x) for x in graph.input]
    outs = [to_ioinfo(x) for x in graph.output]
    return ins, outs, m

def summarize_ort(session: ort.InferenceSession):
    ins = []
    for i in session.get_inputs():
        ins.append({"name": i.name, "type": i.type, "shape": i.shape})
    outs = []
    for o in session.get_outputs():
        outs.append({"name": o.name, "type": o.type, "shape": o.shape})
    return ins, outs

def model_initializers(m: onnx.ModelProto) -> Dict[str, np.ndarray]:
    return {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}

def build_graph_maps(m: onnx.ModelProto):
    """Maps for backward traversal from outputs."""
    prod = {}  # tensor -> node that PRODUCES it
    consumers = collections.defaultdict(list)  # tensor -> list of nodes that CONSUME it
    for node in m.graph.node:
        for o in node.output:
            prod[o] = node
        for i in node.input:
            consumers[i].append(node)
    return prod, consumers

def bfs_back_from_outputs(m: onnx.ModelProto, limit_nodes: int = 10000):
    """Breadth-first walk backward from graph outputs, returning nodes on those paths.
    Uses a string key (node.name or joined outputs) to make NodeProto 'hashable' for visited-set."""
    prod, _ = build_graph_maps(m)
    outputs = [o.name for o in m.graph.output]
    visited_node_keys = set()
    visited_tensors = set(outputs)
    queue = collections.deque(outputs)
    order = []

    while queue and len(visited_node_keys) < limit_nodes:
        t = queue.popleft()
        node = prod.get(t)
        if node is None:
            continue
        node_key = node.name if node.name else "|".join(node.output)  # stable key
        if node_key in visited_node_keys:
            continue
        visited_node_keys.add(node_key)
        order.append(node)
        for i in node.input:
            if i not in visited_tensors:
                visited_tensors.add(i)
                queue.append(i)
    return order


def scan_graph_hints(m: onnx.ModelProto):
    inits = model_initializers(m)
    hints = {
        "constants_count": len(inits),
        "div_255_like": [],
        "mul_1000_like": [],
        "mul_0p001_like": [],
        "mean_std_like": [],
        "root_subtract_pattern": [],
        "notes": [],
    }

    # Simple constant value scans
    for name, arr in inits.items():
        if arr.size == 1:
            v = safe_float(arr)
            if v is None: continue
            if nearly(v, [1/255.0, 255.0]): hints["div_255_like"].append({"name": name, "value": v})
            if nearly(v, [1000.0]): hints["mul_1000_like"].append({"name": name, "value": v})
            if nearly(v, [0.001]): hints["mul_0p001_like"].append({"name": name, "value": v})
        if arr.shape in [(3,), (1,3), (1,3,1,1)]:
            if np.all(np.isfinite(arr)):
                # Not perfect, but plausible mean/std or per-channel affine
                hints["mean_std_like"].append({"name": name, "shape": list(arr.shape), "sample": [float(x) for x in arr.reshape(-1)[:3]]})

    # Look for patterns near outputs (Mul by ~1000 etc.)
    order = bfs_back_from_outputs(m)
    for node in order:
        if node.op_type == "Mul":
            # If any input is a scalar initializer close to 1000/0.001
            for inp in node.input:
                if inp in inits and inits[inp].size == 1:
                    v = safe_float(inits[inp])
                    if v is None: continue
                    if nearly(v, [1000.0]):
                        hints["notes"].append(f"Final-path Mul by ~1000 at node '{node.name or '(no-name)'}'.")
                    if nearly(v, [0.001]):
                        hints["notes"].append(f"Final-path Mul by ~0.001 at node '{node.name or '(no-name)'}'.")

    # Try to catch "Subtract root joint" pattern anywhere:
    #   Sub( X, Gather(X, indices=[root]) ) or Sub( X, root_tensor_broadcast )
    # This is heuristic because axes are not explicit here.
    prod, _ = build_graph_maps(m)
    for node in m.graph.node:
        if node.op_type == "Sub":
            gather_like = None
            base_like = None
            for inp in node.input:
                p = prod.get(inp)
                if p and p.op_type in ("Gather","GatherND","Slice"):
                    gather_like = p
                else:
                    base_like = inp
            if gather_like is not None:
                # Is the gather index a known small int (pelvis/root)? check initializer
                for g_in in gather_like.input:
                    if g_in in inits and inits[g_in].size >= 1:
                        idxs = inits[g_in].reshape(-1)
                        idxs_int = [int(x) for x in idxs[: min(4, idxs.size)]]
                        hints["root_subtract_pattern"].append({
                            "sub_node": node.name or "(no-name)",
                            "gather_node": gather_like.name or "(no-name)",
                            "gather_first_indices": idxs_int
                        })
                        break

    return hints

# ---------- Synthetic Probes ----------

def make_base_sequence(T: int, J: int, scale: float = 200.0, jitter: float = 2.0, seed=0):
    """
    Make a simple standing figure in the XY plane around (0,0), with score=1.
    'scale' controls overall body height in input coordinates.
    """
    rng = np.random.RandomState(seed)
    # Start with a vertical line skeleton: y from -scale/2 .. +scale/2, x ~ 0
    y = np.linspace(-0.45*scale, 0.55*scale, J)
    x = np.zeros(J)
    score = np.ones(J)
    base = np.stack([x, y, score], axis=-1)  # (J,3)
    # Temporal stack with light jitter
    seq = base[None, ...] + rng.randn(T, J, 3) * (jitter * np.array([0.5, 1.0, 0.0])[None, None, :])
    return seq  # (T,J,3)

def translate_seq(seq, dx, dy):
    out = seq.copy()
    out[..., 0] += dx
    out[..., 1] += dy
    return out

def scale_seq(seq, s):
    out = seq.copy()
    out[..., :2] *= s
    return out

def run_model(sess: ort.InferenceSession, input_name: str, x: np.ndarray):
    assert x.ndim == 4, f"Expected (B,T,J,3), got {x.shape}"
    return sess.run(None, {input_name: x})

def median_bone_length_3d(pts: np.ndarray, edges: List[Tuple[int,int]]):
    # pts: (T,J,3)
    # return median over all frames and bones
    if pts.ndim == 4:  # (B,T,J,3)
        pts = pts[0]
    lengths = []
    for (a,b) in edges:
        v = pts[:, a, :] - pts[:, b, :]
        d = np.linalg.norm(v, axis=-1)  # (T,)
        lengths.append(d)
    if not lengths:
        return np.nan
    all_d = np.concatenate(lengths, axis=0)
    return float(np.median(all_d))

def pelvis_center_distance(pts: np.ndarray, root_index: int):
    if pts.ndim == 4:
        pts = pts[0]
    pel = pts[:, root_index, :]   # (T,3)
    d = np.linalg.norm(pel, axis=-1)  # distance from origin each frame
    return float(np.median(d)), float(np.mean(d)), float(np.std(d))

def compare_pose(pts_a: np.ndarray, pts_b: np.ndarray):
    # Return median per-joint, per-frame L2 difference
    if pts_a.ndim == 4: pts_a = pts_a[0]
    if pts_b.ndim == 4: pts_b = pts_b[0]
    d = np.linalg.norm(pts_a - pts_b, axis=-1)  # (T,J)
    return float(np.median(d)), float(np.mean(d))

def infer_units(median_bone_len: float):
    """
    Heuristic:
      - meters if ~0.1 .. 3.5
      - millimeters if ~100 .. 3500
      - unknown otherwise
    """
    if not np.isfinite(median_bone_len):
        return "unknown", 0.0, "non-finite median bone length"
    m_range = (0.1, 3.5)
    mm_range = (100.0, 3500.0)
    if m_range[0] <= median_bone_len <= m_range[1]:
        # map length 0.1 -> conf 0.5, 1.7 -> conf 1.0, 3.5 -> conf 0.6 (soft)
        center = 1.7
        conf = max(0.5, 1.0 - abs(median_bone_len - center)/center)
        return "meters", float(conf), f"median_bone_length={median_bone_len:.3f} in [{m_range[0]}, {m_range[1]}]"
    if mm_range[0] <= median_bone_len <= mm_range[1]:
        center = 1700.0
        conf = max(0.5, 1.0 - abs(median_bone_len - center)/center)
        return "millimeters", float(conf), f"median_bone_length={median_bone_len:.1f} in [{mm_range[0]}, {mm_range[1]}]"
    return "unknown", 0.1, f"median_bone_length={median_bone_len:.3f} outside typical ranges"

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to MotionBERT 3D .onnx")
    ap.add_argument("--skeleton", required=True, help="Path to skeletons.json")
    ap.add_argument("--skeleton-name", default="auto", help="Which skeleton def to use (or 'auto')")
    ap.add_argument("--report", default="report.json", help="Where to write the JSON report")
    ap.add_argument("--frames", type=int, default=16, help="Temporal length T for probes")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # 1) I/O from ONNX
    print("== ONNX I/O ==")
    inputs, outputs, model = get_io_info(args.model)
    for io in inputs:
        print(f"[INPUT]  name={io.name}  np_dtype={io.np_dtype}  onnx_dtype={io.onnx_dtype}  shape={io.shape}")
    for io in outputs:
        print(f"[OUTPUT] name={io.name}  np_dtype={io.np_dtype}  onnx_dtype={io.onnx_dtype}  shape={io.shape}")

    # 2) ORT session and concrete shapes
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    ort_ins, ort_outs = summarize_ort(sess)
    print("\n== ORT meta ==")
    print("ORT inputs:", ort_ins)
    print("ORT outputs:", ort_outs)

    # Decide concrete dims for probes
    in0 = sess.get_inputs()[0]
    assert len(in0.shape) == 4, f"Expected rank-4 input, got {in0.shape}"
    B = 1
    T = args.frames if isinstance(in0.shape[1], str) else int(in0.shape[1])
    J = 17 if isinstance(in0.shape[2], str) else int(in0.shape[2])
    C = 3
    if isinstance(in0.shape[0], int) and in0.shape[0] not in (1, 0):
        B = int(in0.shape[0])
    print(f"\nUsing concrete probe shape: (B={B}, T={T}, J={J}, C={C})")

    # 3) Graph scan hints
    print("\n== Graph scan (constants & patterns) ==")
    hints = scan_graph_hints(model)
    print(json.dumps(hints, indent=2, default=float))

    # 4) Skeleton
    sk_all = load_skeletons(args.skeleton)
    sk = pick_skeleton(sk_all, args.skeleton_name, J)
    print(f"\n== Skeleton: '{sk.name}' (root_index={sk.root_index}, joints={len(sk.joints)}) ==")

    # 5) Probes
    print("\n== Probing model behavior ==")
    base_seq = make_base_sequence(T=T, J=J, scale=200.0, jitter=2.0, seed=args.seed)
    zero_seq = np.zeros_like(base_seq); zero_seq[..., 2] = 1.0  # score=1
    trans_seq = translate_seq(base_seq, dx=400.0, dy=-250.0)
    scale2_seq = scale_seq(base_seq, 2.0)

    # Pack batch dim
    base_in = base_seq[None, ...].astype(np.float32)
    zero_in = zero_seq[None, ...].astype(np.float32)
    trans_in = trans_seq[None, ...].astype(np.float32)
    scale2_in = scale2_seq[None, ...].astype(np.float32)

    # Ensure last channel "score" is 1.0
    for arr in (base_in, zero_in, trans_in, scale2_in):
        arr[..., 2] = 1.0

    # Try a safe baseline run (zeros) to stabilize unknown preproc
    try:
        out_zero = run_model(sess, in0.name, zero_in)
    except Exception as e:
        print("[FATAL] Could not run model with zeros:", e)
        sys.exit(1)

    out_names = [o.name for o in sess.get_outputs()]
    outs_zero = [np.asarray(o) for o in out_zero]
    # Pick the first output as 3D pose if ambiguous
    P0 = outs_zero[0]
    print(f"Zero-run output[0] shape={P0.shape}, dtype={P0.dtype}, stats: min={np.nanmin(P0):.4f}, max={np.nanmax(P0):.4f}, mean={np.nanmean(P0):.4f}")

    # Now run with base/trans/scale2
    P_base = np.asarray(run_model(sess, in0.name, base_in)[0])
    P_trans = np.asarray(run_model(sess, in0.name, trans_in)[0])
    P_scale2 = np.asarray(run_model(sess, in0.name, scale2_in)[0])

    # Ensure expected last-dim=3
    assert P_base.shape[-1] == 3, f"Output last dim is not 3: {P_base.shape}"

    # 6) Measurements
    med_bone = median_bone_length_3d(P_base, sk.edges)
    pelvis_med, pelvis_mean, pelvis_std = pelvis_center_distance(P_base, sk.root_index)
    diff_trans_med, diff_trans_mean = compare_pose(P_base, P_trans)
    diff_scale_med, diff_scale_mean = compare_pose(P_base, P_scale2)

    # Translation invariance relative to skeleton scale (robust)
    # If translating input by a big (dx,dy) barely changes output → pelvis-centered or normalized
    # Normalize differences by a robust output scale (median bone length)
    norm = med_bone if np.isfinite(med_bone) and med_bone > 0 else max(1.0, stable_percentile(np.abs(P_base), 90))
    trans_invariance = 1.0 - min(1.0, (diff_trans_med / (norm + 1e-9)))
    scale_invariance = 1.0 - min(1.0, (diff_scale_med / (norm + 1e-9)))

    # crude confidence from multiple signals
    pelvis_centered = pelvis_med / (norm + 1e-9) < 0.08 and trans_invariance > 0.6
    scale_normalized = scale_invariance > 0.5

    units_guess, units_conf, units_note = infer_units(med_bone)

    report = {
        "model_io": {
            "inputs": [{"name": io.name, "dtype": io.np_dtype, "shape": io.shape} for io in inputs],
            "outputs": [{"name": io.name, "dtype": io.np_dtype, "shape": io.shape} for io in outputs],
            "ort_inputs": ort_ins,
            "ort_outputs": ort_outs,
        },
        "graph_hints": hints,
        "skeleton_used": {
            "name": sk.name,
            "root_index": sk.root_index,
            "joint_count": len(sk.joints),
        },
        "probes": {
            "output_zero_stats": {
                "min": float(np.nanmin(P0)),
                "max": float(np.nanmax(P0)),
                "mean": float(np.nanmean(P0)),
            },
            "median_bone_length": med_bone,
            "pelvis_center_distance": {
                "median": pelvis_med, "mean": pelvis_mean, "std": pelvis_std
            },
            "compare_base_vs_translated": {
                "median_L2": diff_trans_med, "mean_L2": diff_trans_mean
            },
            "compare_base_vs_scaled_x2": {
                "median_L2": diff_scale_med, "mean_L2": diff_scale_mean
            },
            "normalizer": norm,
            "translation_invariance_score": trans_invariance,
            "scale_invariance_score": scale_invariance,
        },
        "inferred": {
            "pelvis_centered": {
                "value": bool(pelvis_centered),
                "confidence": float(min(0.99, 0.5 + 0.5*trans_invariance)),
                "evidence": f"pelvis_median/norm={pelvis_med/(norm+1e-9):.3f}, trans_inv={trans_invariance:.3f}"
            },
            "scale_normalization": {
                "value": bool(scale_normalized),
                "confidence": float(min(0.99, 0.5 + 0.5*scale_invariance)),
                "evidence": f"scale_inv={scale_invariance:.3f} (x2 input test)"
            },
            "units": {
                "value": units_guess,
                "confidence": units_conf,
                "evidence": units_note
            },
        },
    }

    print("\n== Inference / Behavior Summary ==")
    print(json.dumps(report["inferred"], indent=2))

    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote report to {args.report}")

if __name__ == "__main__":
    main()
