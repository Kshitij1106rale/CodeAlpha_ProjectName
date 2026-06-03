"""
SORT (Simple Online and Realtime Tracking) implementation.
Uses a Kalman Filter for state estimation and the Hungarian algorithm for assignment.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(boxA, boxB):
    """
    Computes Intersection over Union (IoU) between two bounding boxes.
    Bounding box format: [x1, y1, x2, y2]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    unionArea = float(boxAArea + boxBArea - interArea)
    if unionArea <= 0:
        return 0.0

    return interArea / unionArea


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box [x1, y1, x2, y2] and returns [cx, cy, s, r]^T
    where s = area, r = aspect ratio (w/h).
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h) if h > 0 else 1.0
    return np.array([cx, cy, s, r], dtype=np.float64).reshape((4, 1))


def convert_x_to_bbox(x):
    """
    Takes state vector [cx, cy, s, r, ...]^T and returns [x1, y1, x2, y2].
    """
    cx = x[0, 0]
    cy = x[1, 0]
    s = max(x[2, 0], 1.0)       # clamp area to be positive
    r = max(x[3, 0], 0.01)      # clamp aspect ratio to be positive

    w = np.sqrt(s * r)
    h = s / w if w > 0 else 0.0

    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return np.array([x1, y1, x2, y2]).reshape((1, 4))


class KalmanBoxTracker:
    """Tracks a single object using a 7-dimensional Kalman Filter."""
    count = 0

    def __init__(self, bbox, class_id=0, score=1.0):
        """Initializes a tracker using initial bounding box [x1,y1,x2,y2]."""

        # State transition matrix (constant velocity model)
        # State: [cx, cy, s, r, v_cx, v_cy, v_s]
        self.F = np.eye(7, dtype=np.float64)
        self.F[0, 4] = 1.0  # cx += v_cx
        self.F[1, 5] = 1.0  # cy += v_cy
        self.F[2, 6] = 1.0  # s  += v_s

        # Measurement matrix
        self.H = np.zeros((4, 7), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        # Measurement noise covariance
        self.R = np.diag([1.0, 1.0, 10.0, 10.0]).astype(np.float64)

        # Process noise covariance
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001]).astype(np.float64)

        # State covariance matrix — high uncertainty for velocities initially
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 10000.0, 10000.0, 10000.0]).astype(np.float64)

        # Initial state
        z = convert_bbox_to_z(bbox)
        self.x = np.zeros((7, 1), dtype=np.float64)
        self.x[:4] = z

        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.class_id = class_id
        self.score = score

        self.hits = 1            # total number of successful matches
        self.hit_streak = 1      # consecutive matches (starts at 1 since we just detected it)
        self.time_since_update = 0
        self.age = 1             # total frames since creation

    def update(self, bbox, class_id=0, score=1.0):
        """Updates the state vector with observed bbox."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.class_id = class_id
        self.score = score

        z = convert_bbox_to_z(bbox)

        # Kalman update step
        y = z - self.H @ self.x                          # innovation
        S = self.H @ self.P @ self.H.T + self.R          # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)         # Kalman gain

        self.x = self.x + K @ y
        I_KH = np.eye(7, dtype=np.float64) - K @ self.H
        self.P = I_KH @ self.P                           # Joseph form simplified

    def predict(self):
        """Advances state and returns predicted bounding box [x1,y1,x2,y2]."""
        # Prevent negative area
        if self.x[2, 0] + self.x[6, 0] <= 0:
            self.x[6, 0] = 0.0

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.age += 1
        self.time_since_update += 1
        self.hit_streak = 0 if self.time_since_update > 0 else self.hit_streak

        return convert_x_to_bbox(self.x)[0]

    def get_state(self):
        """Returns current bounding box estimate [x1,y1,x2,y2]."""
        return convert_x_to_bbox(self.x)[0]


class Sort:
    """SORT: Simple Online and Realtime Tracking."""

    def __init__(self, max_age=7, min_hits=1, iou_threshold=0.25):
        """
        Args:
            max_age: Maximum frames a track can go unmatched before deletion.
            min_hits: Minimum consecutive hits before a track is displayed.
            iou_threshold: Minimum IoU to accept a detection-to-track match.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 6))):
        """
        Args:
            dets: numpy array of detections [[x1,y1,x2,y2,score,class_id], ...]
        Returns:
            numpy array of tracks [[x1,y1,x2,y2,track_id,class_id,score], ...]
        """
        self.frame_count += 1

        # ── Predict existing trackers ─────────────────────────────────
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            if np.any(np.isnan(pos)):
                to_del.append(t)

        # Remove invalid trackers (reverse order to preserve indices)
        for t in sorted(to_del, reverse=True):
            self.trackers.pop(t)

        # ── Associate detections to trackers ──────────────────────────
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, self.trackers, self.iou_threshold
        )

        # ── Update matched trackers with assigned detections ──────────
        for m in matched:
            self.trackers[m[1]].update(
                dets[m[0], :4],
                class_id=int(dets[m[0], 5]),
                score=dets[m[0], 4],
            )

        # ── Create new trackers for unmatched detections ──────────────
        for i in unmatched_dets:
            trk = KalmanBoxTracker(
                dets[i, :4],
                class_id=int(dets[i, 5]),
                score=dets[i, 4],
            )
            self.trackers.append(trk)

        # ── Collect results and prune dead tracks ─────────────────────
        ret = []
        trackers_to_keep = []
        for trk in self.trackers:
            d = trk.get_state()

            # Return if recently updated AND (enough hits OR still in warm-up)
            is_recent = trk.time_since_update < 1
            has_enough_hits = trk.hit_streak >= self.min_hits
            in_warmup = self.frame_count <= self.min_hits

            if is_recent and (has_enough_hits or in_warmup):
                ret.append(np.concatenate((d, [trk.id, trk.class_id, trk.score])))

            # Keep track if it hasn't exceeded max_age
            if trk.time_since_update <= self.max_age:
                trackers_to_keep.append(trk)

        # Replace list cleanly (no mutation during iteration)
        self.trackers = trackers_to_keep

        if len(ret) > 0:
            return np.stack(ret)
        return np.empty((0, 7))


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.25):
    """
    Assigns detections to tracked objects using the Hungarian algorithm.
    Returns (matches, unmatched_detections, unmatched_trackers).
    """
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0,), dtype=int),
        )

    if len(detections) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.empty((0,), dtype=int),
            np.arange(len(trackers)),
        )

    # Build IoU cost matrix
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float64)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            # Only match same class
            if int(det[5]) == int(trk.class_id):
                iou_matrix[d, t] = iou(det[:4], trk.get_state())

    # Solve assignment (maximize IoU = minimize negative IoU)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    # Build matched/unmatched sets
    matched_indices = set()
    matches = []
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= iou_threshold:
            matches.append([r, c])
            matched_indices.add(('d', r))
            matched_indices.add(('t', c))

    unmatched_dets = [d for d in range(len(detections)) if ('d', d) not in matched_indices]
    unmatched_trks = [t for t in range(len(trackers)) if ('t', t) not in matched_indices]

    if len(matches) > 0:
        matches = np.array(matches, dtype=int)
    else:
        matches = np.empty((0, 2), dtype=int)

    return matches, np.array(unmatched_dets), np.array(unmatched_trks)
