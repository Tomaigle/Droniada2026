import pyrealsense2 as rs  # type: ignore
import numpy as np
import cv2
import time
from ultralytics import YOLO  # type: ignore

CONFIDENCE_THRESHOLD = 0.7
MODEL_PATH = "/home/bartek/Studia/Orion/realsense/best.pt"  # swap to yolov8s/m/l/x for more accuracy
SHOW_DEPTH_MAP = True  # toggle depth visualization side by side
SHOW_FPS = True

# ── YOLO ─────────────────────────────────────────────────────────────────────
model = YOLO(MODEL_PATH)

# ── RealSense ─────────────────────────────────────────────────────────────────
pipeline = rs.pipeline()  # type: ignore
config = rs.config()  # type: ignore

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # type: ignore
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # type: ignore

align = rs.align(rs.stream.color)  # type: ignore
profile = pipeline.start(config)

# Get depth scale (converts raw depth units → meters)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Colorizer for depth visualization
colorizer = rs.colorizer()  # type: ignore
colorizer.set_option(rs.option.color_scheme, 2)  # type: ignore  # 2 = WhiteToBlack

# ── FPS counter ───────────────────────────────────────────────────────────────

fps_time = time.time()
fps_value = 0.0
frame_count = 0

print("Running — press Q to quit, D to toggle depth map")

try:
    while True:
        # ── Grab frames ──────────────────────────────────────────────────────
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # ── YOLO inference ───────────────────────────────────────────────────
        results = model(color_image, verbose=False)

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < CONFIDENCE_THRESHOLD:
                    continue

                cls = int(box.cls[0])
                label = model.names[cls]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # ── Depth at object center ───────────────────────────────────
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Average over a 10x10 patch for stability
                patch = depth_image[max(0, cy - 5) : cy + 5, max(0, cx - 5) : cx + 5]
                valid = patch[patch > 0]
                depth_m = (
                    float(np.median(valid)) * depth_scale if valid.size > 0 else 0.0
                )

                # ── Draw ─────────────────────────────────────────────────────
                # Color box by distance: green=close, yellow=mid, red=far
                if depth_m < 1.0:
                    box_color = (0, 255, 0)  # green
                elif depth_m < 2.5:
                    box_color = (0, 200, 255)  # yellow
                else:
                    box_color = (0, 0, 255)  # red

                cv2.rectangle(color_image, (x1, y1), (x2, y2), box_color, 2)

                # Label background for readability
                text = f"{label} {conf:.0%}"
                depth_text = f"{depth_m:.2f}m" if depth_m > 0 else "depth N/A"
                cv2.rectangle(color_image, (x1, y1 - 36), (x1 + 160, y1), box_color, -1)
                cv2.putText(
                    color_image,
                    text,
                    (x1 + 4, y1 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                )
                cv2.putText(
                    color_image,
                    depth_text,
                    (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                )

                # Dot at depth sample point
                cv2.circle(color_image, (cx, cy), 4, (255, 255, 255), -1)

        # ── FPS ──────────────────────────────────────────────────────────────
        frame_count += 1
        if frame_count % 15 == 0:
            fps_value = 15 / (time.time() - fps_time)
            fps_time = time.time()

        if SHOW_FPS:
            cv2.putText(
                color_image,
                f"FPS: {fps_value:.1f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

        # ── Depth map (optional side panel) ──────────────────────────────────
        if SHOW_DEPTH_MAP:
            depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            display = np.hstack((color_image, depth_colormap))
        else:
            display = color_image

        cv2.imshow("RealSense + YOLO", display)

        # ── Key handling ─────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d"):
            SHOW_DEPTH_MAP = not SHOW_DEPTH_MAP

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
