"""
Ball Tracker — RealSense + MAVLink (MatekF405-WTE via UART on RPi3)
====================================================================
- Connects to ArduPilot flight controller over UART (/dev/serial0)
- Live RealSense camera feed with ball detection overlay
- Prints MAVLink commands to terminal in real time
- Closes servo on S5 (output 9 → RC channel 9 / SERVO9) on grip detection
- Releases servo 3 seconds after grip

Wiring:
    RPi3 TX (GPIO14/pin8)  → MatekF405-WTE RX (UART)
    RPi3 RX (GPIO15/pin10) → MatekF405-WTE TX (UART)
    GND ↔ GND

ArduPilot setup:
    SERIALx_PROTOCOL = 2  (MAVLink 2) on the UART connected to RPi
    SERIALx_BAUD     = 57 (57600) — adjust to match BAUD below
    SERVO9_FUNCTION  = 0  (passthrough/manual) — S5 = servo output 9
    RC9_OPTION       = 0  or leave unconfigured

Requirements:
    pip install ultralytics opencv-python numpy pyrealsense2 pymavlink

Run:
    python findmeballs.py
"""

import os

os.environ["DISPLAY"] = ":0"
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import numpy as np
import time
import threading
from ultralytics import YOLO
import pyrealsense2 as rs
from pymavlink import mavutil

# ── CONFIG ────────────────────────────────────────────────
MODEL_PATH = "yolov8n.pt"  # auto-downloads if missing
GRIP_DISTANCE = 1.00  # [m] grip when closer than this
CENTER_THRESH = 0.16  # [m] acceptable XY error before descending
BALL_CLASS_ID = 32  # coco: sports ball
CONF_THRESHOLD = 0.05
KP = 0.5  # proportional gain for speed commands
MAX_SPEED = 0.5  # [m/s] max sent to drone

# MAVLink / UART
UART_PORT = "/dev/serial0"  # RPi3 primary UART (disable BT if needed)
UART_BAUD = 57600
MAVLINK_SYSTEM_ID = 1  # ArduPilot system ID (default)
MAVLINK_COMP_ID = 1  # autopilot component

# Servo (S5 = MAIN OUT 9 on MatekF405-WTE → MAVLink servo output index 9)
SERVO_CHANNEL = 9  # 1-based servo output number
SERVO_CLOSED_PWM = 2000  # PWM µs when gripper is CLOSED
SERVO_OPEN_PWM = 1000  # PWM µs when gripper is OPEN
GRIP_HOLD_S = 3.0  # seconds to hold grip before releasing
# ─────────────────────────────────────────────────────────


# ── MAVLink connection ────────────────────────────────────
def connect_mavlink(port: str, baud: int) -> mavutil.mavfile:
    print(f"Connecting to ArduPilot on {port} @ {baud}...")
    mav = mavutil.mavlink_connection(
        port, baud=baud, source_system=255, source_component=0
    )
    mav.wait_heartbeat(timeout=10)
    print(
        f"Heartbeat received — system {mav.target_system} "
        f"component {mav.target_component}"
    )
    return mav


def set_servo(mav: mavutil.mavfile, channel: int, pwm: int) -> None:
    """Send MAV_CMD_DO_SET_SERVO to move a servo output."""
    mav.mav.command_long_send(
        mav.target_system,
        mav.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
        0,  # confirmation
        channel,  # param1 — servo number (1-based)
        pwm,  # param2 — PWM µs
        0,
        0,
        0,
        0,
        0,
    )


# ── RealSense ─────────────────────────────────────────────
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


def get_depth(depth_frame, cx, cy, w, h) -> float:
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


# ── Control ───────────────────────────────────────────────
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

    if centered and depth > GRIP_DISTANCE:
        vz = float(np.clip(KP * (depth - GRIP_DISTANCE), 0, MAX_SPEED))
    else:
        vz = 0.0

    grip = centered and depth < GRIP_DISTANCE
    return vx, vy, vz, grip


# ── Terminal output ───────────────────────────────────────
def print_mavlink(vx, vy, vz, grip, depth, err_x, err_y, servo_state):
    ts = time.strftime("%H:%M:%S")
    servo_info = f"  SERVO={'CLOSED' if servo_state else 'OPEN'}"
    if grip:
        print(
            f"[{ts}]  *** GRIP ***  depth={depth:.2f}m"
            f"  err=({err_x:+.3f}, {err_y:+.3f})m{servo_info}"
        )
    else:
        print(
            f"[{ts}]  MOVE  vx={vx:+.2f}  vy={vy:+.2f}  vz={vz:+.2f} m/s"
            f"  |  depth={depth:.2f}m  err=({err_x:+.3f}, {err_y:+.3f})m{servo_info}"
        )


# ── Overlay ───────────────────────────────────────────────
def draw_overlay(frame, cx, cy, fw, fh, depth, vx, vy, vz, grip, servo_closed):
    fc = (fw // 2, fh // 2)
    col = (0, 255, 0) if grip else (0, 200, 255)

    cv2.circle(frame, (cx, cy), 10, col, 2)
    cv2.line(frame, (cx - 14, cy), (cx + 14, cy), col, 2)
    cv2.line(frame, (cx, cy - 14), (cx, cy + 14), col, 2)
    cv2.arrowedLine(frame, fc, (cx, cy), col, 2, tipLength=0.2)

    cv2.line(frame, (fc[0] - 20, fc[1]), (fc[0] + 20, fc[1]), (255, 255, 255), 1)
    cv2.line(frame, (fc[0], fc[1] - 20), (fc[0], fc[1] + 20), (255, 255, 255), 1)

    cv2.putText(
        frame,
        f"{depth:.2f}m",
        (cx + 14, cy - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        col,
        2,
        cv2.LINE_AA,
    )

    if grip:
        txt = "MAVLink: GRIP"
        c = (0, 255, 0)
    else:
        txt = f"MAVLink: vx={vx:+.2f} vy={vy:+.2f} vz={vz:+.2f} m/s"
        c = (0, 200, 255)
    cv2.putText(frame, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, c, 2, cv2.LINE_AA)

    servo_txt = f"SERVO S5: {'CLOSED' if servo_closed else 'OPEN'}"
    servo_col = (0, 255, 0) if servo_closed else (80, 80, 255)
    cv2.putText(
        frame,
        servo_txt,
        (10, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        servo_col,
        2,
        cv2.LINE_AA,
    )


# ── Main ──────────────────────────────────────────────────
def main():
    print("Loading YOLOv8...")
    model = YOLO(MODEL_PATH)

    print("Starting RealSense...")
    pipeline, align, fx, fy = setup_realsense()

    mav = connect_mavlink(UART_PORT, UART_BAUD)

    # Ensure servo starts open
    set_servo(mav, SERVO_CHANNEL, SERVO_OPEN_PWM)
    print(f"Servo S5 (channel {SERVO_CHANNEL}) initialised → OPEN ({SERVO_OPEN_PWM}µs)")

    fw, fh = 640, 480
    fc_x, fc_y = fw // 2, fh // 2

    print("\n=== Live Feed Started — press Q to quit ===\n")
    print(f"{'TIME':10}  COMMAND")
    print("-" * 72)

    last_print = 0.0
    servo_closed = False  # current gripper state
    grip_time = None  # when grip was first triggered

    # Background thread: keep MAVLink heartbeat alive so ArduPilot doesn't
    # time-out the connection (optional but good practice).
    def heartbeat_loop():
        while True:
            mav.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_GCS,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0,
                0,
                0,
            )
            time.sleep(1)

    hb_thread = threading.Thread(target=heartbeat_loop, daemon=True)
    hb_thread.start()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())

            # ── Detection ──────────────────────────────────────────────────
            results = model(frame, verbose=False, classes=[BALL_CLASS_ID])[0]

            best = None
            for box in results.boxes:
                conf = float(box.conf[0])
                if conf < CONF_THRESHOLD:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx_b = (x1 + x2) // 2
                cy_b = (y1 + y2) // 2
                depth = get_depth(depth_frame, cx_b, cy_b, fw, fh)
                if best is None or depth < best[2]:
                    best = (cx_b, cy_b, depth, x1, y1, x2, y2)

            # ── Servo release timer ────────────────────────────────────────
            if servo_closed and grip_time is not None:
                if time.time() - grip_time >= GRIP_HOLD_S:
                    set_servo(mav, SERVO_CHANNEL, SERVO_OPEN_PWM)
                    servo_closed = False
                    grip_time = None
                    print(f"[{time.strftime('%H:%M:%S')}]  SERVO RELEASED → OPEN")

            # ── Commands + draw ────────────────────────────────────────────
            if best:
                cx_b, cy_b, depth, x1, y1, x2, y2 = best

                err_x_m = (cx_b - fc_x) * depth / fx
                err_y_m = (cy_b - fc_y) * depth / fy

                vx, vy, vz, grip = compute_mavlink_commands(err_x_m, err_y_m, depth)

                # Close servo on fresh grip (don't retrigger while held)
                if grip and not servo_closed:
                    set_servo(mav, SERVO_CHANNEL, SERVO_CLOSED_PWM)
                    servo_closed = True
                    grip_time = time.time()
                    print(
                        f"[{time.strftime('%H:%M:%S')}]  SERVO CLOSED "
                        f"→ releasing in {GRIP_HOLD_S:.0f}s"
                    )

                now = time.time()
                if now - last_print > 0.1:
                    print_mavlink(
                        vx, vy, vz, grip, depth, err_x_m, err_y_m, servo_closed
                    )
                    last_print = now

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 1)
                draw_overlay(
                    frame, cx_b, cy_b, fw, fh, depth, vx, vy, vz, grip, servo_closed
                )

            else:
                now = time.time()
                if now - last_print > 0.5:
                    print(
                        f"[{time.strftime('%H:%M:%S')}]  NO BALL — HOVER  "
                        f"vx=0 vy=0 vz=0  SERVO={'CLOSED' if servo_closed else 'OPEN'}"
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

                servo_txt = f"SERVO S5: {'CLOSED' if servo_closed else 'OPEN'}"
                servo_col = (0, 255, 0) if servo_closed else (80, 80, 255)
                cv2.putText(
                    frame,
                    servo_txt,
                    (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    servo_col,
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Ball Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Safe shutdown — open servo, stop camera
        set_servo(mav, SERVO_CHANNEL, SERVO_OPEN_PWM)
        pipeline.stop()
        cv2.destroyAllWindows()
        print("\nStopped — servo released.")


if __name__ == "__main__":
    main()
