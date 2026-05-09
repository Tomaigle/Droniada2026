import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from dataclasses import dataclass
from typing import Optional
import logging
import math
import config

log = logging.getLogger(__name__)


@dataclass
class Detection:
    colour: str
    cx: int
    cy: int
    depth: float
    conf: float
    bbox: tuple
    err_x_m: float = 0.0
    err_y_m: float = 0.0


def _pixel_err_to_world(
    px_err_x: float, px_err_y: float, depth_ray: float, fx: float, fy: float
) -> tuple[float, float]:
    tilt = math.radians(config.CAMERA_TILT_DEG)
    xc = px_err_x * depth_ray / fx
    yc = px_err_y * depth_ray / fy
    zc = depth_ray

    body_forward = zc * math.cos(tilt) - yc * math.sin(tilt)
    body_right = xc

    err_forward = body_forward - config.GRIPPER_OFFSET_X_M
    err_right = body_right - config.GRIPPER_OFFSET_Y_M
    return err_forward, err_right


class RealsenseCamera:
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
            rs.stream_color, self.width, self.height, rs.format.bgr8, config.FPS
        )
        cfg.enable_stream(
            rs.stream_depth, self.width, self.height, rs.format.z16, config.FPS
        )
        profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream_color)

        intr = rs.video_stream_profile(
            profile.get_stream(rs.stream.color)
        ).get_intrinsics()
        self.fx, self.fy = intr.fx, intr.fy

        log.info("Realsense initialized")

    def stop(self) -> None:
        self.pipeline.stop()

    def get_frames(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=2000)
            aligned = self.align.process(frames)
            color_f = aligned.get_color_frame()
            depth_f = aligned.get_depth_frame()
            if not color_f or not depth_f:
                return None, None
            return np.asanyarray(color_f.get_data()), np.asanyarray(depth_f.get_data())
        except RuntimeError as e:
            log.warning("Frame grab failed", e)

    def get_depth_at(self, depth_frame, cx: int, cy: int) -> float:
        samples = []
        for dx in (-3, 0, 3):
            for dy in (-3, 0, 3):
                x = int(np.clip(cx + dx, 0, self.width - 1))
                y = int(np.clip(cy + dy, 0, self.height - 1))
                d = depth_frame.get_distance(x, y)
                if d > 0.05:
                    samples.append(d)
        return float(np.median(samples)) if samples else 0.0


class ObjectDetector:
    def __init__(self, camera: RealsenseCamera):
        self.cam = camera
        log.info(f"Loading YOLO {config.YOLO_PATH}")
        self.model = YOLO(config.YOLO_PATH)
        self._all_classes = list(config.BALL_CLASSES.values()) + [
            config.BARREL_CLASS_ID
        ]
        self._cache: dict[int, tuple[list[Detection], list[Detection]]] = {}

    def clear_cache(self) -> None:
        self._cache.clear()

    def _run(
        self, frame: np.ndarray, depth_frame
    ) -> tuple[list[Detection], list[Detection]]:
        fid = id(frame)
        if fid in self._cache:
            return self._cache[fid]

        fc_x = self.cam.width // 2
        fc_y = self.cam.height // 2

        results = self.model(frame, verbose=False, classes=self._all_classes)[0]
        balls: list[Detection] = []
        barrels: list[Detection] = []

        for box in results.boxes:
            conf = float(box.conf[0])
            cid = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            depth = self.cam.get_depth_at(depth_frame, cx, cy)
            if depth <= 0.0:
                continue

            err_x, err_y = _pixel_err_to_world(
                cx - fc_x, cy - fc_y, depth, self.cam.fx, self.cam.fy
            )

            if cid == config.BARREL_CLASS_ID:
                if conf < config.BARREL_CONF_THRESHOLD:
                    continue
                barrels.append(
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
            else:
                if conf < config.BALL_CONF_THRESHOLD:
                    continue
                colour = next(
                    (k for k, v in config.BALL_CLASSES.items() if v == cid), "unknown"
                )
                balls.append(
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

        balls.sort(key=lambda d: d.depth)
        barrels.sort(key=lambda d: d.depth)
        self._cache[fid] = (balls, barrels)
        return balls, barrels

    def detect_balls(
        self,
        frame: np.ndarray,
        depth_frame,
        target_colour: Optional[str] = None,
    ) -> list[Detection]:
        balls, _ = self._run(frame, depth_frame)
        if target_colour:
            balls = [d for d in balls if d.colour == target_colour]
        return balls

    def detect_barrels(self, frame: np.ndarray, depth_frame) -> list[Detection]:
        _, barrels = self._run(frame, depth_frame)
        return barrels

    def detect_all(self, frame: np.ndarray, depth_frame) -> list[Detection]:
        balls, barrels = self._run(frame, depth_frame)
        combined = balls + barrels
        combined.sort(key=lambda d: d.depth)
        return combined


class BallDetector:
    def __init__(self, detector: ObjectDetector):
        self._d = detector

    def detect(self, frame, depth_frame, target_colour=None) -> list[Detection]:
        return self._d.detect_balls(frame, depth_frame, target_colour)


class BarrelDetector:
    def __init__(self, detector: ObjectDetector):
        self._d = detector

    def detect(self, frame, depth_frame) -> list[Detection]:
        return self._d.detect_barrels(frame, depth_frame)
