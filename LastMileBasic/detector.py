"""
detector.py — RealSense camera setup and YOLO-based object detection.

Provides:
  - RealSenseCamera: frame grabber with aligned depth
  - BallDetector:    detects coloured tennis balls, returns best target
  - BarrelDetector:  detects blue barrels for drop targeting

When USE_STOCK_MODEL=True the detector treats all sports-ball detections
as "unknown colour" and falls back to HSV colour classification.
When USE_STOCK_MODEL=False it reads per-class colour from BALL_CLASSES.
"""

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from dataclasses import dataclass
from typing import Optional
import logging
import config

log = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Detection:
    colour: str          # "blue" | "red" | "yellow" | "barrel" | "unknown"
    cx: int              # pixel centre x
    cy: int              # pixel centre y
    depth: float         # metres to object centre
    conf: float          # detection confidence
    bbox: tuple          # (x1, y1, x2, y2)
    err_x_m: float = 0.0  # metres right of frame centre
    err_y_m: float = 0.0  # metres below frame centre


# ── RealSense camera ──────────────────────────────────────────────────────────

class RealSenseCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.align = None
        self.fx = self.fy = 0.0
        self.width  = config.FRAME_WIDTH
        self.height = config.FRAME_HEIGHT

    def start(self) -> None:
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, config.FPS)
        cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16,  config.FPS)
        profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)

        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        intr = depth_profile.get_intrinsics()
        self.fx, self.fy = intr.fx, intr.fy
        log.info("RealSense started — fx=%.1f fy=%.1f", self.fx, self.fy)

    def stop(self) -> None:
        self.pipeline.stop()

    def get_frames(self) -> tuple[Optional[np.ndarray], Optional[object]]:
        """Returns (color_frame_ndarray, depth_frame) or (None, None) on drop."""
        frames = self.pipeline.wait_for_frames(timeout_ms=2000)
        aligned = self.align.process(frames)
        color_f = aligned.get_color_frame()
        depth_f = aligned.get_depth_frame()
        if not color_f or not depth_f:
            return None, None
        return np.asanyarray(color_f.get_data()), depth_f

    @staticmethod
    def get_depth_at(depth_frame, cx: int, cy: int, w: int, h: int) -> float:
        """Robust depth: median of 3×3 sample grid around (cx, cy)."""
        samples = []
        for dx in (-3, 0, 3):
            for dy in (-3, 0, 3):
                x = int(np.clip(cx + dx, 0, w - 1))
                y = int(np.clip(cy + dy, 0, h - 1))
                d = depth_frame.get_distance(x, y)
                if d > 0.05:
                    samples.append(d)
        return float(np.median(samples)) if samples else 0.0


# ── HSV colour classifier (stock model fallback) ──────────────────────────────

# Hue ranges (0–179 in OpenCV)
_HSV_RANGES = {
    "blue":   [(100, 60, 60), (130, 255, 255)],
    "red":    [(0,  80, 80),  (10,  255, 255)],   # red wraps — also check 160-179
    "red2":   [(160, 80, 80), (179, 255, 255)],
    "yellow": [(20, 100, 100),(35,  255, 255)],
}

def _classify_colour_hsv(roi_bgr: np.ndarray) -> str:
    """Classify dominant colour of a small BGR crop via HSV histogram."""
    if roi_bgr.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    best, best_pct = "unknown", 0.0
    total = hsv.shape[0] * hsv.shape[1]
    for name, (lo, hi) in [
        ("blue",   (_HSV_RANGES["blue"][0],   _HSV_RANGES["blue"][1])),
        ("red",    (_HSV_RANGES["red"][0],    _HSV_RANGES["red"][1])),
        ("yellow", (_HSV_RANGES["yellow"][0], _HSV_RANGES["yellow"][1])),
    ]:
        mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
        if name == "red":
            mask2 = cv2.inRange(hsv, np.array(_HSV_RANGES["red2"][0]),
                                     np.array(_HSV_RANGES["red2"][1]))
            mask = cv2.bitwise_or(mask, mask2)
        pct = cv2.countNonZero(mask) / total
        if pct > best_pct:
            best_pct, best = pct, name
    return best if best_pct > 0.15 else "unknown"


# ── Ball detector ─────────────────────────────────────────────────────────────

class BallDetector:
    def __init__(self, camera: RealSenseCamera):
        self.cam = camera
        log.info("Loading YOLO model: %s", config.MODEL_PATH)
        self.model = YOLO(config.MODEL_PATH)

        # Which class IDs to scan depends on model type
        if config.USE_STOCK_MODEL:
            self._classes = [config.STOCK_BALL_CLASS_ID]
        else:
            self._classes = list(config.BALL_CLASSES.values())

    def detect(
        self,
        frame: np.ndarray,
        depth_frame,
        target_colour: Optional[str] = None,
    ) -> list[Detection]:
        """
        Run detection on frame.
        If target_colour is given, return only detections of that colour.
        Returns list sorted by depth ascending (nearest first).
        """
        fc_x = self.cam.width  // 2
        fc_y = self.cam.height // 2

        results = self.model(frame, verbose=False, classes=self._classes)[0]
        detections: list[Detection] = []

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < config.CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            depth = self.cam.get_depth_at(depth_frame, cx, cy, self.cam.width, self.cam.height)

            # Colour determination
            if config.USE_STOCK_MODEL:
                roi = frame[max(0, y1):y2, max(0, x1):x2]
                colour = _classify_colour_hsv(roi)
            else:
                class_id = int(box.cls[0])
                colour = next(
                    (k for k, v in config.BALL_CLASSES.items() if v == class_id),
                    "unknown"
                )

            if target_colour and colour != target_colour:
                continue

            err_x_m = (cx - fc_x) * depth / self.cam.fx if self.cam.fx else 0.0
            err_y_m = (cy - fc_y) * depth / self.cam.fy if self.cam.fy else 0.0

            detections.append(Detection(
                colour=colour, cx=cx, cy=cy, depth=depth, conf=conf,
                bbox=(x1, y1, x2, y2), err_x_m=err_x_m, err_y_m=err_y_m,
            ))

        detections.sort(key=lambda d: d.depth)
        return detections


# ── Barrel detector ───────────────────────────────────────────────────────────

# Barrels are blue cylinders — detect by HSV colour segmentation + contour.
# No separate YOLO class needed for basic stage; upgrade to trained class later.

class BarrelDetector:
    """
    Detects blue barrels via HSV colour segmentation.
    Returns list[Detection] with colour="barrel", sorted nearest first.
    """

    def __init__(self, camera: RealSenseCamera):
        self.cam = camera

    def detect(self, frame: np.ndarray, depth_frame) -> list[Detection]:
        fc_x = self.cam.width  // 2
        fc_y = self.cam.height // 2

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([100, 80, 40]), np.array([130, 255, 200]))

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[Detection] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 800:          # too small — noise
                continue
            x1, y1, w, h = cv2.boundingRect(cnt)
            x2, y2 = x1 + w, y1 + h
            cx = x1 + w // 2
            cy = y1 + h // 2

            # Basic barrel shape check: expect roughly circular top view
            aspect = w / h if h else 0
            if not (0.4 < aspect < 2.5):
                continue

            depth = self.cam.get_depth_at(depth_frame, cx, cy, self.cam.width, self.cam.height)
            if depth <= 0:
                continue

            err_x_m = (cx - fc_x) * depth / self.cam.fx if self.cam.fx else 0.0
            err_y_m = (cy - fc_y) * depth / self.cam.fy if self.cam.fy else 0.0
            conf = min(1.0, area / 10000)   # proxy confidence from area

            if conf < config.BARREL_CONF:
                continue

            detections.append(Detection(
                colour="barrel", cx=cx, cy=cy, depth=depth, conf=conf,
                bbox=(x1, y1, x2, y2), err_x_m=err_x_m, err_y_m=err_y_m,
            ))

        detections.sort(key=lambda d: d.depth)
        return detections
