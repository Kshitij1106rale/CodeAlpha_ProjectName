"""
Real-Time Object Detection and Tracking
========================================
Uses YOLOv8n (ONNX) for detection via OpenCV DNN and a custom SORT tracker.

Controls:
  ESC / q  - Quit
  SPACE    - Pause / Resume
  t        - Toggle tracking on/off
  b        - Toggle bounding box style (solid / dashed)
  i        - Toggle info panel
"""

import cv2
import numpy as np
import os
import sys
import time

from sort_tracker import Sort

# ── COCO class names (80 classes) ──────────────────────────────────────────────
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# ── Distinct colors for tracking IDs ──────────────────────────────────────────
def _generate_colors(n=200):
    """Generate n visually distinct colors using HSV."""
    colors = []
    for i in range(n):
        hue = int(i * 180 / n) % 180
        col = np.array([[[hue, 220, 220]]], dtype=np.uint8)
        bgr = cv2.cvtColor(col, cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors

TRACK_COLORS = _generate_colors()


def get_color(track_id):
    """Return a consistent color for a given track ID."""
    return TRACK_COLORS[int(track_id) % len(TRACK_COLORS)]


# ── YOLOv8 Detection ─────────────────────────────────────────────────────────

def letterbox(frame, target_size=640):
    """
    Resize image preserving aspect ratio, pad with gray (114) to target_size.
    Returns: (letterboxed_image, scale, pad_x, pad_y)
    """
    h, w = frame.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create padded image filled with gray (114 is standard YOLO padding)
    padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    return padded, scale, pad_x, pad_y


class YOLOv8Detector:
    """Runs YOLOv8 inference using OpenCV DNN on an ONNX model."""

    def __init__(self, model_path, input_size=640, conf_thresh=0.35, nms_thresh=0.45):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}\n"
                "Run `python download_model.py` first."
            )
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        # Prefer GPU if available
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("[INFO] Using CUDA backend")
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("[INFO] Using CPU backend")

    def detect(self, frame):
        """
        Run detection on a BGR frame using proper letterbox preprocessing.
        Returns numpy array of shape (N, 6): [x1, y1, x2, y2, score, class_id]
        """
        h, w = frame.shape[:2]

        # ── Letterbox preprocessing (preserves aspect ratio) ──────────
        letterboxed, scale, pad_x, pad_y = letterbox(frame, self.input_size)

        # Convert to blob (already correct size, no resize needed)
        blob = cv2.dnn.blobFromImage(
            letterboxed, scalefactor=1 / 255.0,
            size=(self.input_size, self.input_size),
            swapRB=True, crop=False,
        )
        self.net.setInput(blob)
        outputs = self.net.forward()  # shape: (1, 84, 8400)

        # Transpose to (8400, 84)
        preds = outputs[0].T  # (8400, 84)

        # Columns: cx, cy, w, h, then 80 class scores
        boxes_xywh = preds[:, :4]
        scores_all = preds[:, 4:]  # (8400, 80)

        class_ids = np.argmax(scores_all, axis=1)
        confidences = scores_all[np.arange(len(scores_all)), class_ids]

        # Filter by confidence
        mask = confidences > self.conf_thresh
        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes_xywh) == 0:
            return np.empty((0, 6))

        # ── Map coordinates from letterboxed space → original image ───
        # Model outputs are in 640x640 letterbox space
        # Step 1: Remove padding offset
        cx = boxes_xywh[:, 0] - pad_x
        cy = boxes_xywh[:, 1] - pad_y
        bw = boxes_xywh[:, 2]
        bh = boxes_xywh[:, 3]

        # Step 2: Undo the letterbox scale
        cx = cx / scale
        cy = cy / scale
        bw = bw / scale
        bh = bh / scale

        # Step 3: Convert center-wh to corner format
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # Clamp to image boundaries
        x1 = np.clip(x1, 0, w)
        y1 = np.clip(y1, 0, h)
        x2 = np.clip(x2, 0, w)
        y2 = np.clip(y2, 0, h)

        # NMS via OpenCV (expects [x, y, width, height] format)
        nms_boxes = np.stack([x1, y1, bw, bh], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(
            nms_boxes, confidences.tolist(),
            self.conf_thresh, self.nms_thresh,
        )

        if len(indices) == 0:
            return np.empty((0, 6))

        indices = np.array(indices).flatten()
        detections = np.stack([
            x1[indices], y1[indices], x2[indices], y2[indices],
            confidences[indices], class_ids[indices].astype(float),
        ], axis=1)

        return detections


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_detections(frame, detections):
    """Draw raw detections (without tracking) as white boxes."""
    for det in detections:
        x1, y1, x2, y2, score, cls_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label = f"{COCO_CLASSES[int(cls_id)]} {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def draw_tracks(frame, tracks):
    """Draw tracked objects with coloured boxes and IDs."""
    for trk in tracks:
        x1, y1, x2, y2 = int(trk[0]), int(trk[1]), int(trk[2]), int(trk[3])
        track_id = int(trk[4])
        cls_id = int(trk[5])
        score = trk[6]
        color = get_color(track_id)

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        label = f"ID:{track_id} {COCO_CLASSES[cls_id]} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Corner accents
        corner_len = max(1, min(20, (x2 - x1) // 3, (y2 - y1) // 3))
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 3)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 3)
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 3)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 3)
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 3)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 3)
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 3)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 3)


def draw_info_panel(frame, fps, num_detections, num_tracks, tracking_on, paused):
    """Draw a semi-transparent info panel in the top-left corner."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    panel_h = 150
    panel_w = 280
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    y = 35
    cv2.putText(frame, "Object Detection & Tracking", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
    y += 28
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 1)
    y += 24
    cv2.putText(frame, f"Detections: {num_detections}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
    y += 24
    cv2.putText(frame, f"Active Tracks: {num_tracks}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
    y += 24
    status = "ON" if tracking_on else "OFF"
    color = (0, 255, 0) if tracking_on else (0, 0, 255)
    cv2.putText(frame, f"Tracking: {status}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1)

    if paused:
        cv2.putText(frame, "PAUSED", (w // 2 - 60, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)


def draw_controls_bar(frame):
    """Draw a thin controls hint bar at the bottom."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 30), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    text = "ESC/q: Quit | SPACE: Pause | t: Toggle Tracking | i: Toggle Info"
    cv2.putText(frame, text, (10, h - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Resolve model path ────────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "yolov8n.onnx")

    # ── Resolve video source ──────────────────────────────────────────────
    if len(sys.argv) > 1:
        source = sys.argv[1]
        # If a numeric string, treat as camera index
        if source.isdigit():
            source = int(source)
        elif not os.path.isfile(source):
            print(f"[ERROR] Video file not found: {source}")
            sys.exit(1)
    else:
        source = 0  # default webcam

    print(f"[INFO] Video source: {source}")
    print(f"[INFO] Model: {model_path}")

    # ── Initialize detector and tracker ───────────────────────────────────
    detector = YOLOv8Detector(model_path)
    tracker = Sort(max_age=7, min_hits=1, iou_threshold=0.25)

    # ── Open video ────────────────────────────────────────────────────────
    cap = None
    if isinstance(source, int):
        # Try default (MSMF) backend first, then DirectShow as fallback
        for backend_name, backend in [("Default (MSMF)", cv2.CAP_MSMF), ("DirectShow", cv2.CAP_DSHOW)]:
            print(f"[INFO] Trying {backend_name} backend...")
            cap = cv2.VideoCapture(source, backend)
            if cap.isOpened():
                # Warm up — try to get a real frame
                got_frame = False
                for attempt in range(30):
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        print(f"[INFO] Camera ready via {backend_name} after {attempt + 1} attempt(s).")
                        got_frame = True
                        break
                    time.sleep(0.1)
                if got_frame:
                    break
                else:
                    print(f"[INFO] {backend_name} opened but no frames received. Trying next...")
                    cap.release()
                    cap = None
            else:
                print(f"[INFO] {backend_name} failed to open.")
                cap = None
    else:
        cap = cv2.VideoCapture(source)

    if cap is None or not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        sys.exit(1)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"[INFO] Resolution: {frame_w}x{frame_h} @ {vid_fps:.0f} FPS")

    # ── State variables ───────────────────────────────────────────────────
    tracking_on = True
    show_info = True
    paused = False
    fps = 0.0
    frame_count = 0

    window_name = "Object Detection & Tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(frame_w, 1280), min(frame_h, 720))

    print("[INFO] Starting... Press ESC or 'q' to quit.")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret or frame is None:
                # If reading from a video file, loop it
                if isinstance(source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    # Webcam occasionally drops frames — retry instead of quitting
                    time.sleep(0.03)
                    continue

            t0 = time.perf_counter()

            # ── Detection ─────────────────────────────────────────────
            detections = detector.detect(frame)

            # ── Tracking ──────────────────────────────────────────────
            if tracking_on and len(detections) > 0:
                tracks = tracker.update(detections)
            elif tracking_on:
                tracks = tracker.update()
            else:
                tracks = np.empty((0, 7))

            # ── Draw ──────────────────────────────────────────────────
            if tracking_on:
                draw_tracks(frame, tracks)
            else:
                draw_detections(frame, detections)

            if show_info:
                draw_info_panel(
                    frame, fps, len(detections),
                    len(tracks) if tracking_on else 0,
                    tracking_on, paused,
                )
            draw_controls_bar(frame)

            dt = time.perf_counter() - t0
            instant_fps = 1.0 / dt if dt > 0 else 0
            fps = 0.9 * fps + 0.1 * instant_fps  # exponential smoothing
            frame_count += 1
            display_frame = frame
        else:
            # While paused, just redraw the last frame with PAUSED overlay
            display_frame = frame.copy()
            if show_info:
                draw_info_panel(
                    display_frame, fps, 0, 0, tracking_on, paused
                )
            draw_controls_bar(display_frame)

        cv2.imshow(window_name, display_frame)

        # ── Key handling ──────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('t'):
            tracking_on = not tracking_on
            print(f"[INFO] Tracking {'ON' if tracking_on else 'OFF'}")
        elif key == ord('i'):
            show_info = not show_info

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Processed {frame_count} frames. Exiting.")


if __name__ == "__main__":
    main()
