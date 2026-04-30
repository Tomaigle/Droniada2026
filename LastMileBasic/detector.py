"""
detector.py — RealSense camera + YOLO ball detection + HSV barrel detection.

Camera tilt compensation
------------------------
The RealSense is bolted at a fixed downward angle (CAMERA_TILT_DEG) so it
can see both the gripper zone below and objects ahead when in hover.

When aligned over a ball, the camera sees it at a pixel offset from centre
that depends on:
  1. The tilt angle
  2. The depth to the ball
  3. The focal length

We compensate by rotating the pixel error vector by the tilt angle before
converting to real-world XY velocity commands. The result is the horizontal
displacement in the world frame regardless of tilt.

Gripper offset
--------------
The gripper is physically offset from the camera. GRIPPER_OFFSET_X_M and
GRIPPER_OFFSET_Y_M are subtracted from the computed error so the drone
centres the *gripper* over the ball, not the camera.
"""

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from dataclasses import dataclass
from typing import Optional
import logging
import math
import config

log = logging.getLogger(__name__)


# ── Data class ────────────────────────────────────────────────────────────────


@dataclass
class Detection:
    colour: str  # "blue" | "red" | "yellow" | "barrel" | "unknown"
    cx: int  # pixel centre x
    cy: int  # pixel centre y
    depth: float  # metres to object (along camera ray)
    conf: float
    bbox: tuple  # (x1, y1, x2, y2)
    # World-frame error from gripper to object (metres)
    err_x_m: float = 0.0  # positive = object is to the right
    err_y_m: float = 0.0  # positive = object is forward/below


# ── Tilt compensation ──────────────────────────────────────────────────────────


def _pixel_error_to_world(
    px_err_x: float,
    px_err_y: float,
    depth: float,
    fx: float,
    fy: float,
) -> tuple[float, float]:
    """
    Convert pixel error (from frame centre) to world-plane XY error in metres,
    accounting for camera tilt and gripper offset.

    The camera is tilted downward by CAMERA_TILT_DEG. A pixel offset along Y
    in the image corresponds to a mix of forward (world-Y) and vertical
    displacement. We rotate the metric error by the tilt angle to recover
    the horizontal world-plane component.

    Returns (err_x_m, err_y_m) — horizontal displacement of target from gripper.
    """
    tilt_rad = math.radians(config.CAMERA_TILT_DEG)

    # Raw metric errors in camera frame
    raw_x = px_err_x * depth / fx  # camera X = world X (no lateral tilt)
    raw_y = px_err_y * depth / fy  # camera Y mixes world-Y and world-Z

    # Rotate camera-Y component back to world horizontal plane
    world_x = raw_x
    world_y = raw_y * math.cos(tilt_rad)  # horizontal component only

    # Subtract gripper offset so we centre the gripper, not the camera
    world_x -= config.GRIPPER_OFFSET_X_M
    world_y -= config.GRIPPER_OFFSET_Y_M

    return world_x, world_y


# ── RealSense camera ──────────────────────────────────────────────────────────


class RealSenseCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.align = None
        self.fx = 0.0
        self.fy = 0.0
        self.width = config.FRAME_WIDTH
        self.height = config.FRAME_HEIGHT

    def start(self) -> None:
        cfg = rs.config()
        cfg.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, config.FPS
        )
        cfg.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, config.FPS
        )
        profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)
        intr = rs.video_stream_profile(
            profile.get_stream(rs.stream.depth)
        ).get_intrinsics()
        self.fx, self.fy = intr.fx, intr.fy
        log.info(
            "RealSense started — fx=%.1f fy=%.1f tilt=%.1f°",
            self.fx,
            self.fy,
            config.CAMERA_TILT_DEG,
        )

    def stop(self) -> None:
        self.pipeline.stop()

    def get_frames(self) -> tuple[Optional[np.ndarray], Optional[object]]:
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=2000)
            aligned = self.align.process(frames)
            color_f = aligned.get_color_frame()
            depth_f = aligned.get_depth_frame()
            if not color_f or not depth_f:
                return None, None
            return np.asanyarray(color_f.get_data()), depth_f
        except RuntimeError as e:
            log.warning("Frame grab failed: %s", e)
            return None, None

    def get_depth_at(self, depth_frame, cx: int, cy: int) -> float:
        """Robust depth: median of 3×3 sample grid."""
        samples = []
        for dx in (-3, 0, 3):
            for dy in (-3, 0, 3):
                x = int(np.clip(cx + dx, 0, self.width - 1))
                y = int(np.clip(cy + dy, 0, self.height - 1))
                d = depth_frame.get_distance(x, y)
                if d > 0.05:
                    samples.append(d)
        return float(np.median(samples)) if samples else 0.0


# ── HSV colour classifier ──────────────────────────────────────────────────────

_HSV_RANGES = {
    "blue": [(100, 60, 60), (130, 255, 255)],
    "red": [(0, 80, 80), (10, 255, 255)],
    "red2": [(160, 80, 80), (179, 255, 255)],
    "yellow": [(20, 100, 100), (35, 255, 255)],
}


def _classify_colour_hsv(roi_bgr: np.ndarray) -> str:
    if roi_bgr.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    total = roi_bgr.shape[0] * roi_bgr.shape[1]
    best, best_pct = "unknown", 0.0
    for name in ("blue", "red", "yellow"):
        lo, hi = _HSV_RANGES[name]
        mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
        if name == "red":  # red wraps hue
            lo2, hi2 = _HSV_RANGES["red2"]
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lo2), np.array(hi2)))
        pct = cv2.countNonZero(mask) / total
        if pct > best_pct:
            best_pct, best = pct, name
    return best if best_pct > 0.15 else "unknown"


# ── Ball detector ──────────────────────────────────────────────────────────────


class BallDetector:
    def __init__(self, camera: RealSenseCamera):
        self.cam = camera
        log.info("Loading YOLO: %s", config.MODEL_PATH)
        self.model = YOLO(config.MODEL_PATH)
        self._classes = (
            [config.STOCK_BALL_CLASS_ID]
            if config.USE_STOCK_MODEL
            else list(config.BALL_CLASSES.values())
        )

    def detect(
        self,
        frame: np.ndarray,
        depth_frame,
        target_colour: Optional[str] = None,
    ) -> list[Detection]:
        """
        Returns detections sorted nearest-first.
        If target_colour given, filters to that colour only.
        """
        fc_x = self.cam.width // 2
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
            depth = self.cam.get_depth_at(depth_frame, cx, cy)

            if config.USE_STOCK_MODEL:
                roi = frame[max(0, y1) : y2, max(0, x1) : x2]
                colour = _classify_colour_hsv(roi)
            else:
                cid = int(box.cls[0])
                colour = next(
                    (k for k, v in config.BALL_CLASSES.items() if v == cid), "unknown"
                )

            if target_colour and colour != target_colour:
                continue

            err_x, err_y = _pixel_error_to_world(
                cx - fc_x, cy - fc_y, depth, self.cam.fx, self.cam.fy
            )
            detections.append(
                Detection(
                    colour=colour,
                    cx=cx,
                    cy=cy,
                    depth=depth,
                    conf=conf,
                    bbox=(x1, y1, x2, y2),
                    err_x_m=err_x,
                    err_y_m=err_y,
                )
            )

        detections.sort(key=lambda d: d.depth)
        return detections


# ── Barrel detector ────────────────────────────────────────────────────────────


class BarrelDetector:
    """HSV-based blue barrel detection. No YOLO class needed for BASIC stage."""

    def __init__(self, camera: RealSenseCamera):
        self.cam = camera

    def detect(self, frame: np.ndarray, depth_frame) -> list[Detection]:
        fc_x = self.cam.width // 2
        fc_y = self.cam.height // 2

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([100, 80, 40]), np.array([130, 255, 200]))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[Detection] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 800:
                continue
            x1, y1, w, h = cv2.boundingRect(cnt)
            x2, y2 = x1 + w, y1 + h
            cx, cy = x1 + w // 2, y1 + h // 2
            if not (0.4 < (w / h if h else 0) < 2.5):
                continue

            depth = self.cam.get_depth_at(depth_frame, cx, cy)
            if depth <= 0:
                continue

            err_x, err_y = _pixel_error_to_world(
                cx - fc_x, cy - fc_y, depth, self.cam.fx, self.cam.fy
            )
            conf = min(1.0, area / 10000)
            if conf < config.BARREL_CONF:
                continue

            detections.append(
                Detection(
                    colour="barrel",
                    cx=cx,
                    cy=cy,
                    depth=depth,
                    conf=conf,
                    bbox=(x1, y1, x2, y2),
                    err_x_m=err_x,
                    err_y_m=err_y,
                )
            )

        detections.sort(key=lambda d: d.depth)
        return detections
