import os

os.environ["DISPLAY"] = ":0"
os.environ["QT_QPA_PLATFORM"] = "xcb"
import cv2
import numpy as np
import time
from ultralytics import YOLO
import pyrealsense2 as rs

# ── CONFIG ────────────────────────────────────────────────
MODEL_PATH = "color.pt"  # auto-downloads if missing
GRIP_DISTANCE = 1.00  # [m] grip when closer than this
CENTER_THRESH = 0.16  # [m] acceptable XY error before descending
CONF_THRESHOLD = 0.05
KP = 0.5  # proportional gain for speed commands
MAX_SPEED = 0.5  # [m/s] max sent to drone
# ─────────────────────────────────────────────────────────


def setup_realsense():
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(cfg)
    align = rs.align(rs.stream.color)

    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    intr = depth_profile.get_intrinsics()
    fx, fy = intr.fx, intr.fy

    return pipeline, align, fx, fy


def get_depth(depth_frame, cx, cy, w, h):
    """Average depth over small region around point."""
    samples = []
    for dx in [-3, 0, 3]:
        for dy in [-3, 0, 3]:
            x = int(np.clip(cx + dx, 0, w - 1))
            y = int(np.clip(cy + dy, 0, h - 1))
            d = depth_frame.get_distance(x, y)
            if d > 0.05:
                samples.append(d)
    return float(np.median(samples)) if samples else 0.0


def compute_mavlink_commands(err_x_m, err_y_m, depth):
    """
    Returns velocity commands [m/s] and grip flag.
    err_x_m: ball is this far RIGHT of center (positive = right)
    err_y_m: ball is this far BELOW center  (positive = forward)
    depth:   distance to ball [m]
    """
    centered = abs(err_x_m) < CENTER_THRESH and abs(err_y_m) < CENTER_THRESH

    vx = float(np.clip(KP * err_x_m, -MAX_SPEED, MAX_SPEED))
    vy = float(np.clip(KP * err_y_m, -MAX_SPEED, MAX_SPEED))

    # Only descend when centered
    if centered and depth > GRIP_DISTANCE:
        vz = float(np.clip(KP * (depth - GRIP_DISTANCE), 0, MAX_SPEED))
    else:
        vz = 0.0

    grip = centered and depth < GRIP_DISTANCE

    return vx, vy, vz, grip


def print_mavlink(vx, vy, vz, grip, depth, err_x, err_y, class_name, conf):
    ts = time.strftime("%H:%M:%S")
    label = f"class={class_name}({conf:.2f})"
    if grip:
        print(
            f"[{ts}]  *** GRIP ***  {label}  depth={depth:.2f}m  err=({err_x:+.3f}, {err_y:+.3f})m"
        )
    else:
        print(
            f"[{ts}]  MOVE  vx={vx:+.2f}  vy={vy:+.2f}  vz={vz:+.2f} m/s"
            f"  |  {label}  depth={depth:.2f}m  err=({err_x:+.3f}, {err_y:+.3f})m"
        )


def draw_overlay(frame, cx, cy, fw, fh, depth, vx, vy, vz, grip, class_name, conf):
    fc = (fw // 2, fh // 2)

    col = (0, 255, 0) if grip else (0, 200, 255)

    # Ball crosshair
    cv2.circle(frame, (cx, cy), 10, col, 2)
    cv2.line(frame, (cx - 14, cy), (cx + 14, cy), col, 2)
    cv2.line(frame, (cx, cy - 14), (cx, cy + 14), col, 2)

    # Arrow from frame center to ball
    cv2.arrowedLine(frame, fc, (cx, cy), col, 2, tipLength=0.2)

    # Frame center (gripper target)
    cv2.line(frame, (fc[0] - 20, fc[1]), (fc[0] + 20, fc[1]), (255, 255, 255), 1)
    cv2.line(frame, (fc[0], fc[1] - 20), (fc[0], fc[1] + 20), (255, 255, 255), 1)

    # Class + depth label next to ball
    cv2.putText(
        frame,
        f"{class_name} {conf:.2f} {depth:.2f}m",
        (cx + 14, cy - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        col,
        2,
        cv2.LINE_AA,
    )

    # MAVLink command top-left
    if grip:
        txt = f"MAVLink: GRIP [{class_name} {conf:.2f}]"
        c = (0, 255, 0)
    else:
        txt = f"MAVLink: vx={vx:+.2f} vy={vy:+.2f} vz={vz:+.2f} m/s [{class_name} {conf:.2f}]"
        c = (0, 200, 255)

    cv2.putText(frame, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, c, 2, cv2.LINE_AA)


def draw_secondary(frame, x1, y1, x2, y2, class_name, conf):
    """Draw non-target detections in grey."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), (160, 160, 160), 1)
    cv2.putText(
        frame,
        f"{class_name} {conf:.2f}",
        (x1, y1 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )


def main():
    print("Loading YOLOv8...")
    model = YOLO(MODEL_PATH)
    class_names = model.names  # {id: name}

    print("Starting RealSense...")
    pipeline, align, fx, fy = setup_realsense()

    fw, fh = 640, 480
    fc_x, fc_y = fw // 2, fh // 2

    print("\n=== Live Feed Started — press Q to quit ===\n")
    print(f"{'TIME':10}  COMMAND")
    print("-" * 75)

    last_print = 0.0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())

            # ── Detection ──────────────────────────────────
            results = model(frame, verbose=False)[0]

            detections = []
            for box in results.boxes:
                conf = float(box.conf[0])
                if conf < CONF_THRESHOLD:
                    continue
                cls_id = int(box.cls[0])
                cls_name = class_names.get(cls_id, str(cls_id))
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                depth = get_depth(depth_frame, cx, cy, fw, fh)
                detections.append((cx, cy, depth, x1, y1, x2, y2, cls_name, conf))

            # Sort by depth — closest = target
            detections.sort(key=lambda d: d[2])

            # ── Draw secondaries first (behind target) ──────
            for det in detections[1:]:
                _, _, _, x1, y1, x2, y2, cls_name, conf = det
                draw_secondary(frame, x1, y1, x2, y2, cls_name, conf)

            # ── Commands + draw target ─────────────────────
            if detections:
                cx, cy, depth, x1, y1, x2, y2, cls_name, conf = detections[0]

                err_x_m = (cx - fc_x) * depth / fx
                err_y_m = (cy - fc_y) * depth / fy

                vx, vy, vz, grip = compute_mavlink_commands(err_x_m, err_y_m, depth)

                now = time.time()
                if now - last_print > 0.1:
                    print_mavlink(
                        vx, vy, vz, grip, depth, err_x_m, err_y_m, cls_name, conf
                    )
                    last_print = now

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 1)
                draw_overlay(
                    frame, cx, cy, fw, fh, depth, vx, vy, vz, grip, cls_name, conf
                )

            else:
                now = time.time()
                if now - last_print > 0.5:
                    print(
                        f"[{time.strftime('%H:%M:%S')}]  NO BALL — HOVER  vx=0 vy=0 vz=0"
                    )
                    last_print = now

                cv2.putText(
                    frame,
                    "NO BALL — HOVER",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (80, 80, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.line(
                    frame, (fc_x - 20, fc_y), (fc_x + 20, fc_y), (255, 255, 255), 1
                )
                cv2.line(
                    frame, (fc_x, fc_y - 20), (fc_x, fc_y + 20), (255, 255, 255), 1
                )

            cv2.imshow("Ball Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("\nStopped.")


if __name__ == "__main__":
    main()
