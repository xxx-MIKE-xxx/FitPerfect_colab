# run_rgb_to_motionbert3d.py
# Pipeline: YOLOv8 (person) -> RTMPose (2D keypoints, SimCC) -> COCO17->H36M17 -> normalize -> MotionBERT-3D (243)
# References:
# - YOLOv8 ONNX output [1,84,N], 4 box + 80 cls (N=8400); NMS outside the graph. https://github.com/ultralytics/ultralytics/issues/3680
#   Official example also letterboxes & rescales back. https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-OpenCV-ONNX-Python/main.py
#   scale_boxes for letterbox math. https://docs.ultralytics.com/reference/utils/ops/
# - RTMPose SimCC decoding: softmax over (x,y) axis, argmax/expectation; divide by split_ratio (≈2). 
#   https://mmpose.readthedocs.io/en/latest/_modules/mmpose/codecs/simcc_label.html
#   https://github.com/open-mmlab/mmpose/issues/2755
# - MotionBERT "wild" pipeline: H36M-17, per-frame [-1,1] normalization unless --pixel; rootrel option zeros pelvis. 
#   https://github.com/Walter0807/MotionBERT/blob/main/infer_wild.py

import argparse, os, json, math, glob
import numpy as np
import cv2
import onnxruntime as ort

# ---------------------------
# Utilities
# ---------------------------

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize with unchanged aspect ratio using padding, like Ultralytics."""
    shape = img.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (w, h)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    # Return ratio and padding for de-letterboxing later
    return img, r, (dw, dh)

def sigmoid(x): 
    return 1.0 / (1.0 + np.exp(-x))

def xywh2xyxy(x):
    # x=[...,4] format center-x,center-y,width,height -> x1y1x2y2
    y = np.empty_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y

def nms(boxes, scores, iou_thr=0.45, max_det=300):
    """Basic NMS on xyxy boxes."""
    if len(boxes) == 0:
        return []
    boxes = boxes.astype(np.float32)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0 and len(keep) < max_det:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)

        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]
    return keep

def crop_to_aspect(img, box_xyxy, out_hw=(256, 192), scale=1.25):
    """Adjust a box to model aspect ratio, expand, clamp, then crop+resize."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    # expand
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = (x2 - x1) * scale
    bh = (y2 - y1) * scale

    # match aspect ratio
    out_h, out_w = out_hw
    target_ar = out_w / out_h  # 192/256 = 0.75
    if bw / bh > target_ar:
        bh = bw / target_ar
    else:
        bw = bh * target_ar

    x1n = max(0, int(round(cx - bw / 2)))
    y1n = max(0, int(round(cy - bh / 2)))
    x2n = min(w - 1, int(round(cx + bw / 2)))
    y2n = min(h - 1, int(round(cy + bh / 2)))

    crop = img[y1n:y2n, x1n:x2n]
    if crop.size == 0:
        return None, None  # fail-safe

    resized = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    # Keep mapping (rect in original image)
    rect = (x1n, y1n, x2n - x1n, y2n - y1n)  # (x, y, w, h)
    return resized, rect

def simcc_decode(simcc_x, simcc_y, split_ratio=2.0):
    """
    simcc_x: [B, K, Wx]; simcc_y: [B, K, Hy]
    Return coords in the model input space (width=192, height=256) and scores per keypoint.
    """
    # Softmax stabilizes logits before argmax/expectation
    def softmax(a, axis=-1):
        a = a - np.max(a, axis=axis, keepdims=True)
        ea = np.exp(a)
        return ea / np.sum(ea, axis=axis, keepdims=True)

    px = softmax(simcc_x, axis=2)
    py = softmax(simcc_y, axis=2)

    x_idx = np.argmax(px, axis=2)
    y_idx = np.argmax(py, axis=2)

    # Convert to pixel space in the model input
    x = x_idx.astype(np.float32) / float(split_ratio)  # width axis
    y = y_idx.astype(np.float32) / float(split_ratio)  # height axis

    # Confidence = max prob along each axis, take sqrt of product (geometric mean)
    conf = np.sqrt(np.max(px, axis=2) * np.max(py, axis=2)).astype(np.float32)
    coords = np.stack([x, y], axis=-1)  # [B,K,2]
    return coords, conf

def coco17_to_h36m17(coco_xy, coco_conf):
    """
    Rough mapping used in many H36M-from-COCO pipelines (same spirit as MotionBERT 'wild' conversion).
    COCO order: 0 nose,1 leye,2 reye,3 lear,4 rear,5 lsho,6 rsho,7 lelb,8 relb,9 lwri,10 rwri,11 lhip,12 rhip,13 lknee,14 rknee,15 lank,16 rank
    H36M order (17):
      0 Pelvis, 1 RHip, 2 RKnee, 3 RAnkle, 4 LHip, 5 LKnee, 6 LAnkle,
      7 Spine1, 8 Neck, 9 Head, 10 Site(head top),
      11 LShoulder, 12 LElbow, 13 LWrist, 14 RShoulder, 15 RElbow, 16 RWrist
    """
    nose, leye, reye = coco_xy[0], coco_xy[1], coco_xy[2]
    lsho, rsho = coco_xy[5], coco_xy[6]
    lelb, relb = coco_xy[7], coco_xy[8]
    lwri, rwri = coco_xy[9], coco_xy[10]
    lhip, rhip = coco_xy[11], coco_xy[12]
    lknee, rknee = coco_xy[13], coco_xy[14]
    lank, rank = coco_xy[15], coco_xy[16]

    # Midpoints
    pelvis = (lhip + rhip) / 2.0
    neck   = (lsho + rsho) / 2.0
    spine1 = (pelvis + neck) / 2.0
    eyes_mid = (leye + reye) / 2.0 if (leye.any() and reye.any()) else nose

    # Head / Site approximations:
    head = eyes_mid
    site = nose  # "head top" not in COCO; nose is a reasonable proxy in many wild pipelines.

    h36m = np.zeros((17, 2), dtype=np.float32)
    h36m[0]  = pelvis
    h36m[1]  = rhip;   h36m[2]  = rknee;  h36m[3]  = rank
    h36m[4]  = lhip;   h36m[5]  = lknee;  h36m[6]  = lank
    h36m[7]  = spine1; h36m[8]  = neck;   h36m[9]  = head; h36m[10] = site
    h36m[11] = lsho;   h36m[12] = lelb;   h36m[13] = lwri
    h36m[14] = rsho;   h36m[15] = relb;   h36m[16] = rwri

    # Confidence propagate as mean of constituents where needed
    c = coco_conf
    c_pelvis = (c[11] + c[12]) / 2
    c_neck   = (c[5] + c[6]) / 2
    c_spine1 = (c_pelvis + c_neck) / 2
    c_head   = (c[1] + c[2]) / 2 if (c[1] > 0 and c[2] > 0) else c[0]
    c_site   = c[0]

    h36m_conf = np.zeros((17,), dtype=np.float32)
    h36m_conf[0]  = c_pelvis
    h36m_conf[1]  = c[12]; h36m_conf[2]  = c[14]; h36m_conf[3]  = c[16]
    h36m_conf[4]  = c[11]; h36m_conf[5]  = c[13]; h36m_conf[6]  = c[15]
    h36m_conf[7]  = c_spine1; h36m_conf[8] = c_neck; h36m_conf[9] = c_head; h36m_conf[10] = c_site
    h36m_conf[11] = c[5]; h36m_conf[12] = c[7]; h36m_conf[13] = c[9]
    h36m_conf[14] = c[6]; h36m_conf[15] = c[8]; h36m_conf[16] = c[10]
    return h36m, h36m_conf

def normalize_to_minus1_1(xy, img_w, img_h):
    """Per-frame normalization matching MotionBERT non-pixel path (center + min side)."""
    s = min(img_w, img_h) / 2.0
    cx, cy = img_w / 2.0, img_h / 2.0
    xy_norm = xy.copy()
    xy_norm[..., 0] = (xy[..., 0] - cx) / s
    xy_norm[..., 1] = (xy[..., 1] - cy) / s
    return xy_norm

def read_video_frames(path, max_frames=None):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, im = cap.read()
        if not ret: break
        frames.append(im)
        if max_frames and len(frames) >= max_frames: break
    cap.release()
    return frames

# ---------------------------
# Model wrappers
# ---------------------------

class YOLOv8ONNX:
    def __init__(self, model_path, providers=None, conf_thres=0.25, iou_thres=0.45):
        self.session = ort.InferenceSession(model_path, providers=providers or ['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name  # 'images'
        self.conf_thres = conf_thres
        self.iou_thres  = iou_thres

    def detect_person(self, im_bgr):
        im_letter, r, (dw, dh) = letterbox(im_bgr, (640, 640))
        im_rgb = cv2.cvtColor(im_letter, cv2.COLOR_BGR2RGB)
        x = (im_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]  # [1,3,640,640]
        out = self.session.run(None, {self.input_name: x})[0]  # [1,84,8400]
        preds = out[0].T  # [8400,84]

        boxes = preds[:, :4]
        cls_logits = preds[:, 4:]  # 80 classes (COCO)
        cls_probs = sigmoid(cls_logits)
        cls_ids = np.argmax(cls_probs, axis=1)
        cls_scores = cls_probs[np.arange(cls_probs.shape[0]), cls_ids]

        # Keep class 0 (person) only
        m = (cls_ids == 0) & (cls_scores >= self.conf_thres)
        boxes = boxes[m]; scores = cls_scores[m]

        if len(boxes) == 0:
            return None, 0.0

        # Back to original image space from letterboxed 640x640
        # boxes are xywh relative to letterboxed image
        boxes = xywh2xyxy(boxes)
        # undo padding/scale: (x - pad)/r
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / r
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / r
        # clip to image
        h0, w0 = im_bgr.shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w0 - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h0 - 1)

        keep = nms(boxes, scores, self.iou_thres, max_det=50)
        if not keep:
            return None, 0.0

        # Pick highest-score person after NMS
        best = keep[0]
        return boxes[best].astype(np.float32), float(scores[best])

class RTMPoseONNX:
    def __init__(self, model_path, providers=None, simcc_split_ratio=2.0):
        self.session = ort.InferenceSession(model_path, providers=providers or ['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name  # 'input'
        self.simcc_split_ratio = simcc_split_ratio

    # inside class RTMPoseONNX:
    def infer(self, crop_bgr):
        # BGR -> RGB float32
        im = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        # MMPose defaults
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std  = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        im = (im - mean) / std
        # to NCHW
        x = im.transpose(2, 0, 1)[None, ...]   # [1,3,256,192]
        simcc_x, simcc_y = self.session.run(None, {self.input_name: x})
        coords, conf = simcc_decode(simcc_x, simcc_y, split_ratio=self.simcc_split_ratio)
        return coords[0], conf[0]


# ---------------------------
# Main pipeline
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', type=str, required=True, help='Path to input video')
    ap.add_argument('--yolo', type=str, required=True, help='YOLOv8n ONNX file path')
    ap.add_argument('--rtmpose', type=str, required=True, help='RTMPose-m ONNX file path')
    ap.add_argument('--motionbert', type=str, required=True, help='MotionBERT-3D-243 ONNX file path')
    ap.add_argument('--out', type=str, default='mb3d_output.json', help='Where to save the 3D result')
    ap.add_argument('--rootrel', action='store_true', help='Zero pelvis (root-relative) before exporting')
    ap.add_argument('--providers', type=str, nargs='*', default=None, help='ONNXRuntime providers')
    ap.add_argument('--simcc_ratio', type=float, default=2.0, help='RTMPose SimCC split ratio')
    ap.add_argument('--window', type=int, default=243, help='Temporal window for MotionBERT')
    ap.add_argument('--person_conf', type=float, default=0.25)
    ap.add_argument('--person_iou', type=float, default=0.45)
    args = ap.parse_args()

    frames = read_video_frames(args.video)
    if not frames:
        raise SystemExit(f'No frames read from {args.video}')

    yolo = YOLOv8ONNX(args.yolo, providers=args.providers, conf_thres=args.person_conf, iou_thres=args.person_iou)
    rtm  = RTMPoseONNX(args.rtmpose, providers=args.providers, simcc_split_ratio=args.simcc_ratio)

    seq_xyc_h36m = []  # list of [17,3] per frame in IMAGE PIXELS (x,y,conf)

    for idx, im in enumerate(frames):
        prev_bbox = None
        # 1) detect person
        det, score = yolo.detect_person(im)
        if det is None:
            if prev_bbox is None:
                # fallback: full-frame bbox
                h, w = im.shape[:2]
                det = np.array([0, 0, w-1, h-1], dtype=np.float32)
            else:
                det = prev_bbox
        else:
            prev_bbox = det

        # 2) crop to RTMPose aspect, run pose
        crop, rect = crop_to_aspect(im, det, out_hw=(256, 192), scale=1.25)
        if crop is None:
            seq_xyc_h36m.append(seq_xyc_h36m[-1].copy() if seq_xyc_h36m else np.zeros((17,3), np.float32))
            continue

        pose_xy_in, pose_conf = rtm.infer(crop)  # in crop space (width=192, height=256)

        # 3) map back to full image pixels
        rx, ry, rw, rh = rect
        img_w, img_h = im.shape[1], im.shape[0]
        # RTMPose input size is (h=256, w=192)
        x_img = rx + pose_xy_in[:, 0] * (rw / 192.0)
        y_img = ry + pose_xy_in[:, 1] * (rh / 256.0)
        coco_xy = np.stack([x_img, y_img], axis=-1)
        coco_conf = pose_conf

        # 4) COCO-17 -> H36M-17
        h36m_xy, h36m_conf = coco17_to_h36m17(coco_xy, coco_conf)
        h36m_xyc = np.concatenate([h36m_xy, h36m_conf[:, None]], axis=1)  # [17,3]
        seq_xyc_h36m.append(h36m_xyc.astype(np.float32))

    # 5) pack to length 243
    T = args.window
    if len(seq_xyc_h36m) < T:
        # pad by repeating last frame
        last = seq_xyc_h36m[-1]
        seq_xyc_h36m += [last] * (T - len(seq_xyc_h36m))
    elif len(seq_xyc_h36m) > T:
        seq_xyc_h36m = seq_xyc_h36m[:T]

    seq_xyc_h36m = np.stack(seq_xyc_h36m, axis=0)  # [T,17,3]

    # 6) per-frame normalization to [-1,1] (MotionBERT "non-pixel" path)
    #    (if you want pixel mode, comment this block and set pixel=True in MB; ONNX graph has no flag, so we normalize here.)
    im_h, im_w = frames[0].shape[:2]
    xy_norm = normalize_to_minus1_1(seq_xyc_h36m[..., :2], im_w, im_h)
    seq_norm = np.concatenate([xy_norm, seq_xyc_h36m[..., 2:3]], axis=-1)  # [T,17,3]

    # 7) MotionBERT ONNX
    mb_sess = ort.InferenceSession(args.motionbert, providers=args.providers or ['CPUExecutionProvider'])
    mb_in_name = mb_sess.get_inputs()[0].name  # usually 'input'
    mb_out_name = mb_sess.get_outputs()[0].name  # usually 'output'

    mb_in = seq_norm[None, ...].astype(np.float32)  # [1,T,17,3]
    mb_out = mb_sess.run(None, {mb_in_name: mb_in})[0]  # [1,T,17,3]
    X3D = mb_out[0]  # [T,17,3]

    # 8) root-relative option (zero pelvis)
    if args.rootrel:
        X3D[:, 0, :] = 0.0

    # 9) Save
    result = {
        "video": os.path.basename(args.video),
        "T": int(T),
        "h36m_order": ["Pelvis","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle",
                       "Spine1","Neck","Head","Site","LShoulder","LElbow","LWrist",
                       "RShoulder","RElbow","RWrist"],
        "coords_3d": X3D.tolist()  # per-frame [17,3], normalized coords unless you change step 6
    }
    with open(args.out, 'w') as f:
        json.dump(result, f)
    print(f"Saved 3D pose to {args.out}")

if __name__ == "__main__":
    main()

