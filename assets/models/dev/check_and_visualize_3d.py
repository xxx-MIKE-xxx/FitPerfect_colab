#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check & visualize MotionBERT 3D output (Human3.6M-17).

Usage:
  python check_and_visualize_3d.py --json out_3d.json --center --fps 10
"""
import argparse, json, math, os
from dataclasses import dataclass
from typing import Dict, List, Tuple
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

H36M_NAMES = ["Pelvis","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle",
              "Spine1","Neck","Head","Site","LShoulder","LElbow","LWrist",
              "RShoulder","RElbow","RWrist"]

# H36M skeleton edges (for bone length checks)
H36M_EDGES = [
    (0,7), (7,8), (8,9), (9,10),          # spine to head/site
    (8,11), (11,12), (12,13),             # left arm
    (8,14), (14,15), (15,16),             # right arm
    (0,1), (1,2), (2,3),                  # right leg
    (0,4), (4,5), (5,6)                   # left leg
]

RIGHT_LEG = (1,2,3)
LEFT_LEG  = (4,5,6)
RIGHT_ARM = (14,15,16)
LEFT_ARM  = (11,12,13)

@dataclass
class SanityConfig:
    pelvis_tol: float = 0.02
    bone_cv_warn: float = 0.15
    symmetry_warn: float = 0.2
    vel_zscore: float = 6.0
    knee_min_deg: float = 5.0
    knee_max_deg: float = 175.0
    elbow_min_deg: float = 5.0
    elbow_max_deg: float = 175.0

def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    coords = np.asarray(data["coords_3d"], dtype=np.float32)  # [T,17,3]
    return data, coords

def is_finite(arr: np.ndarray) -> bool:
    return np.isfinite(arr).all()

def pelvis_stats(coords: np.ndarray) -> Dict:
    pelvis = coords[:,0,:]
    norms = np.linalg.norm(pelvis, axis=1)
    return {"mean_norm": float(np.mean(norms)), "max_norm": float(np.max(norms)),
            "std_norm": float(np.std(norms)), "frames": int(coords.shape[0])}

def bone_lengths_per_frame(coords: np.ndarray) -> np.ndarray:
    T = coords.shape[0]
    L = np.zeros((T, len(H36M_EDGES)), dtype=np.float32)
    for i,(a,b) in enumerate(H36M_EDGES):
        seg = coords[:,a,:] - coords[:,b,:]
        L[:,i] = np.linalg.norm(seg, axis=1)
    return L

def limb_lengths(coords: np.ndarray, limb: Tuple[int,int,int]) -> Tuple[np.ndarray,np.ndarray]:
    a,b,c = limb
    ab = np.linalg.norm(coords[:,a,:] - coords[:,b,:], axis=1)
    bc = np.linalg.norm(coords[:,b,:] - coords[:,c,:], axis=1)
    return ab, bc

def rel_diff(a: np.ndarray, b: np.ndarray, eps=1e-9) -> np.ndarray:
    return np.abs(a - b) / (np.maximum((a+b)/2.0, eps))

def temporal_velocity(coords: np.ndarray) -> np.ndarray:
    vel = np.linalg.norm(np.diff(coords, axis=0), axis=2)  # [T-1,17]
    return vel

def mad(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)) + 1e-9)

def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba = a - b; bc = c - b
    num = np.sum(ba*bc, axis=1)
    den = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-9
    cosv = np.clip(num/den, -1.0, 1.0)
    return np.degrees(np.arccos(cosv))

def compute_angles(coords: np.ndarray) -> Dict[str, np.ndarray]:
    ang = {}
    ang["right_knee"]  = angle_deg(coords[:,1,:], coords[:,2,:], coords[:,3,:])
    ang["left_knee"]   = angle_deg(coords[:,4,:], coords[:,5,:], coords[:,6,:])
    ang["right_elbow"] = angle_deg(coords[:,14,:], coords[:,15,:], coords[:,16,:])
    ang["left_elbow"]  = angle_deg(coords[:,11,:], coords[:,12,:], coords[:,13,:])
    return ang

def center_pelvis(coords: np.ndarray) -> np.ndarray:
    c = coords.copy()
    c -= c[:,0:1,:]
    return c

def plot_bone_cv(bone_means, bone_stds, out_path):
    cv = (bone_stds / np.maximum(bone_means, 1e-9))
    plt.figure(figsize=(10,4))
    x = np.arange(len(H36M_EDGES))
    plt.bar(x, cv)
    plt.xticks(x, [f"{a}-{b}" for a,b in H36M_EDGES], rotation=60)
    plt.ylabel("Coefficient of variation")
    plt.title("Bone length stability across time")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160); plt.close()
    return cv

def plot_angles(angles: Dict[str,np.ndarray], out_path: str):
    plt.figure(figsize=(10,4))
    for k,v in angles.items():
        plt.plot(v, label=k)
    plt.xlabel("Frame"); plt.ylabel("Angle (deg)")
    plt.title("Knee/Elbow angles over time")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=160); plt.close()

def animate_3d(coords: np.ndarray, out_path: str, fps: int = 10, center: bool=True):
    from matplotlib.animation import PillowWriter
    pts = center_pelvis(coords) if center else coords.copy()
    T = pts.shape[0]
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d')
    # equal aspect
    allmin = pts.min(axis=(0,1)); allmax = pts.max(axis=(0,1))
    rng = (allmax - allmin).max(); mid = (allmax + allmin)/2.0
    ax.set_xlim(mid[0]-rng/2, mid[0]+rng/2)
    ax.set_ylim(mid[1]-rng/2, mid[1]+rng/2)
    ax.set_zlim(mid[2]-rng/2, mid[2]+rng/2)
    scat = ax.scatter([], [], [], s=10)
    lines = [ax.plot([], [], [])[0] for _ in H36M_EDGES]
    ax.set_title("3D pose (pelvis-centered)" if center else "3D pose")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    def init():
        scat._offsets3d = ([], [], [])
        for ln in lines: ln.set_data([], []); ln.set_3d_properties([])
        return [scat] + lines

    def update(t):
        p = pts[t]
        scat._offsets3d = (p[:,0], p[:,1], p[:,2])
        for ln,(a,b) in zip(lines, H36M_EDGES):
            seg = p[[a,b]]
            ln.set_data(seg[:,0], seg[:,1]); ln.set_3d_properties(seg[:,2])
        return [scat] + lines

    ani = animation.FuncAnimation(fig, update, frames=pts.shape[0], init_func=init, blit=True, interval=1000.0/max(fps,1))
    ani.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to out_3d.json")
    ap.add_argument("--fps", type=int, default=10, help="GIF FPS")
    ap.add_argument("--center", action="store_true", help="Pelvis-center the visualization")
    args = ap.parse_args()

    data, coords = load_json(args.json)
    T, J, C = coords.shape
    report = {"file": args.json, "T": int(T), "J": int(J), "C": int(C), "h36m_order": data.get("h36m_order")}

    report["finite"] = bool(is_finite(coords))
    report["has_nan_inf"] = not report["finite"]

    pstats = pelvis_stats(coords)
    report["pelvis"] = pstats
    cfg = SanityConfig()
    report["pelvis_close_to_rootrel"] = pstats["mean_norm"] < cfg.pelvis_tol

    L = bone_lengths_per_frame(coords)
    means = L.mean(axis=0); stds = L.std(axis=0)
    cv = (stds / np.maximum(means, 1e-9))
    report["bone_lengths"] = {"edges": H36M_EDGES, "mean": means.tolist(), "std": stds.tolist(), "cv": cv.tolist()}
    report["bone_cv_flags"] = [i for i,v in enumerate(cv) if v > cfg.bone_cv_warn]

    r_thigh, r_shin = limb_lengths(coords, RIGHT_LEG)
    l_thigh, l_shin = limb_lengths(coords, LEFT_LEG)
    r_uarm,  r_farm = limb_lengths(coords, RIGHT_ARM)
    l_uarm,  l_farm = limb_lengths(coords, LEFT_ARM)

    thigh_diff = rel_diff(r_thigh, l_thigh)
    shin_diff  = rel_diff(r_shin,  l_shin)
    uarm_diff  = rel_diff(r_uarm,  l_uarm)
    farm_diff  = rel_diff(r_farm,  l_farm)

    report["symmetry"] = {
        "mean_rel_diff": {
            "thigh": float(thigh_diff.mean()),
            "shin":  float(shin_diff.mean()),
            "upperarm": float(uarm_diff.mean()),
            "forearm":  float(farm_diff.mean())
        },
        "warn_flags": {
            "thigh": float(thigh_diff.mean()) > cfg.symmetry_warn,
            "shin":  float(shin_diff.mean())  > cfg.symmetry_warn,
            "upperarm": float(uarm_diff.mean()) > cfg.symmetry_warn,
            "forearm":  float(farm_diff.mean()) > cfg.symmetry_warn
        }
    }

    vel = temporal_velocity(coords)
    med = np.median(vel, axis=0)
    madv = np.array([mad(vel[:,j]) for j in range(J)])
    spikes = (vel > (med + cfg.vel_zscore * madv))
    report["temporal"] = {"median_vel_per_joint": med.tolist(),
                           "mad_vel_per_joint": madv.tolist(),
                           "spike_ratio": float(spikes.mean())}

    ang = compute_angles(coords)
    report["angles"] = {
        "right_knee": {"mean": float(np.mean(ang["right_knee"])), "min": float(np.min(ang["right_knee"])), "max": float(np.max(ang["right_knee"]))},
        "left_knee":  {"mean": float(np.mean(ang["left_knee"])),  "min": float(np.min(ang["left_knee"])),  "max": float(np.max(ang["left_knee"]))},
        "right_elbow":{"mean": float(np.mean(ang["right_elbow"])), "min": float(np.min(ang["right_elbow"])), "max": float(np.max(ang["right_elbow"]))},
        "left_elbow": {"mean": float(np.mean(ang["left_elbow"])),  "min": float(np.min(ang["left_elbow"])),  "max": float(np.max(ang["left_elbow"]))}
    }
    def ratio_outside(x, lo, hi):
        return float(np.mean((x < lo) | (x > hi)))
    report["angle_out_of_range_ratio"] = {
        "right_knee": ratio_outside(ang["right_knee"],  cfg.knee_min_deg,  cfg.knee_max_deg),
        "left_knee":  ratio_outside(ang["left_knee"],   cfg.knee_min_deg,  cfg.knee_max_deg),
        "right_elbow":ratio_outside(ang["right_elbow"], cfg.elbow_min_deg, cfg.elbow_max_deg),
        "left_elbow": ratio_outside(ang["left_elbow"],  cfg.elbow_min_deg, cfg.elbow_max_deg),
    }

    out_dir = Path(os.path.dirname(args.json) or ".")
    cv_path   = str(out_dir / "bone_length_cv.png")
    ang_path  = str(out_dir / "knee_elbow_angles.png")
    gif_path  = str(out_dir / "skeleton3d.gif")
    rpt_path  = str(out_dir / "sanity_report.json")

    # plots
    _ = plot_bone_cv(means, stds, cv_path)
    plot_angles(ang, ang_path)
    animate_3d(coords, gif_path, fps=args.fps, center=args.center)

    with open(rpt_path, "w") as f:
        json.dump(report, f, indent=2)

    print("Saved:\n - {}\n - {}\n - {}\n - {}".format(rpt_path, cv_path, ang_path, gif_path))

if __name__ == "__main__":
    main()

