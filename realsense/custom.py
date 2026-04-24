import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import sys

MODEL_PATH = "best.pt"  # <-- your model
CONF_THRESH = 0.5
DEPTH_SAMPLE_RADIUS = 4  # px radius for median depth sampling around center

# ── RealSense pipeline ────────────────────────────────────────────────────────
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipeline.start(config)

# Align depth → color frame
align = rs.align(rs.stream.color)

# Depth scale: converts raw units → meters
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

model = YOLO(MODEL_PATH)


def sample_depth(depth_frame, cx, cy, radius=DEPTH_SAMPLE_RADIUS):
    """Median depth over a small patch — robust to holes/noise."""
    h = depth_frame.get_height()
    w = depth_frame.get_width()
    samples = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            px, py = cx + dx, cy + dy
            if 0 <= px < w and 0 <= py < h:
                d = depth_frame.get_distance(px, py)
                if d > 0:
                    samples.append(d)
    return float(np.median(samples)) if samples else 0.0


try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        h, w = frame.shape[:2]
        frame_cx, frame_cy = w // 2, h // 2

        results = model(frame, conf=CONF_THRESH, verbose=False)[0]

        # Frame center crosshair
        cv2.drawMarker(
            frame, (frame_cx, frame_cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2
        )

        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            bx = (x1 + x2) // 2
            by = (y1 + y2) // 2

            dx = bx - frame_cx
            dy = by - frame_cy
            dist_px = np.hypot(dx, dy)

            # Depth at box center
            depth_m = sample_depth(depth_frame, bx, by)
            depth_str = f"{depth_m:.2f}m" if depth_m > 0 else "N/A"

            # Draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (bx, by), 5, (0, 0, 255), -1)
            cv2.line(frame, (frame_cx, frame_cy), (bx, by), (255, 0, 255), 1)

            hud = f"{label} {conf:.2f} | dx={dx:+d} dy={dy:+d}px | {depth_str}"
            cv2.putText(
                frame, hud, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

            # Terminal
            print(
                f"[Det {i}] {label} ({conf:.2f}) | "
                f"box_center=({bx},{by}) | "
                f"offset dx={dx:+d}px dy={dy:+d}px dist={dist_px:.1f}px | "
                f"depth={depth_str}"
            )

        if results.boxes:
            print("---")

        cv2.imshow("YOLO + RealSense", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
