# Real-Time Object Detection & Tracking

A Python application that performs **real-time object detection** using **YOLOv8n** (ONNX) and **object tracking** using a custom **SORT** (Simple Online and Realtime Tracking) algorithm.

## Features

- **YOLOv8n Detection** — Fast, accurate detection of 80 COCO object classes via OpenCV DNN (no PyTorch needed at runtime)
- **SORT Tracking** — Kalman-filter-based tracker with Hungarian algorithm association, assigning persistent IDs to objects across frames
- **Class-Aware Matching** — Tracking IDs never jump between different object categories
- **Real-Time Display** — Colour-coded bounding boxes, tracking IDs, confidence scores, and an FPS counter
- **Webcam & Video File Support** — Works with any camera or video file

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│  OpenCV      │────▶│  YOLOv8 Detector │────▶│ SORT Tracker │
│  VideoCapture│     │  (cv2.dnn + ONNX)│     │ (Kalman+IoU) │
└─────────────┘     └──────────────────┘     └──────┬───────┘
                                                     │
                                              ┌──────▼───────┐
                                              │  Visualiser   │
                                              │ (boxes, IDs)  │
                                              └──────────────┘
```

## Setup

### 1. Install dependencies

```bash
pip install opencv-python numpy scipy
```

### 2. Download the YOLOv8n ONNX model

```bash
# Option A: If ultralytics is installed
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', opset=12)"

# Option B: Run the download helper (tries multiple URLs)
python download_model.py
```

### 3. Run the application

```bash
# Use default webcam (camera index 0)
python main.py

# Use a specific camera index
python main.py 1

# Use a video file
python main.py path/to/video.mp4
```

## Keyboard Controls

| Key      | Action                  |
|----------|-------------------------|
| `ESC`/`q` | Quit                  |
| `SPACE`  | Pause / Resume          |
| `t`      | Toggle tracking on/off  |
| `i`      | Toggle info panel       |

## Project Structure

| File               | Description                                      |
|--------------------|--------------------------------------------------|
| `main.py`          | Entry point — video loop, detection, visualisation |
| `sort_tracker.py`  | SORT algorithm (Kalman filter + Hungarian matching)|
| `download_model.py`| Helper to download the YOLOv8n ONNX model         |
| `yolov8n.onnx`     | Pre-trained model (generated/downloaded)           |

## How It Works

1. **Video Input** — Each frame is captured from the webcam or video file using `cv2.VideoCapture`.
2. **Detection** — The frame is resized to 640×640 and fed through YOLOv8n via `cv2.dnn`. Raw predictions are parsed and filtered with Non-Maximum Suppression (NMS).
3. **Tracking** — Detections are passed to the SORT tracker, which:
   - Predicts each existing track's new position using a **Kalman Filter**
   - Builds an **IoU cost matrix** between predictions and new detections (class-aware)
   - Solves the assignment problem using the **Hungarian Algorithm** (`scipy.optimize.linear_sum_assignment`)
   - Creates new tracks for unmatched detections and removes stale tracks
4. **Visualisation** — Tracked objects are drawn with colour-coded bounding boxes, unique IDs, class labels, and confidence scores.
