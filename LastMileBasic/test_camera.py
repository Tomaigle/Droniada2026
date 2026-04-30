"""
test_camera.py — Standalone RealSense + detection test.
No MAVLink, no gripper, no mission logic.

Prints detection output to terminal and shows live HUD window.
Use this to:
  - Verify RealSense is working
  - Tune CONF_THRESHOLD and HSV colour ranges
  - Check tilt compensation is correct (err_x/err_y should → 0 when drone
    is centred above a ball)
  - Confirm barrel detection picks up barrels reliably

Run:
    python test_camera.py
    python test_camera.py --no-hud       # terminal only
    python test_camera.py --colour blue  # filter to one colour
    python test_camera.py --barrels      # show barrel detection only
"""

import argparse
import logging
import os
import time

os.environ.setdefault("DISPLAY", ":0")
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
import config
import cv2
from detector import BallDetector, BarrelDetector, RealSenseCamera

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Camera + detection test")
    p.add_argument("--no-hud", action="store_true")
    p.add_argument("--barrels", action="store_true", help="Barrel detection mode")
    p.add_argument("--colour", default=None, help="Filter ball colour: blue/red/yellow")
    return p.parse_args()


def main():
    args = parse_args()
    cam = RealSenseCamera()
    cam.start()

    ball_det = BallDetector(cam)
    barrel_det = BarrelDetector(cam)

    log.info("Camera ready. Press Q to quit.")
    log.info(
        "CAMERA_TILT_DEG=%.1f  GRIPPER_OFFSET=(%.2f, %.2f)",
        config.CAMERA_TILT_DEG,
        config.GRIPPER_OFFSET_X_M,
        config.GRIPPER_OFFSET_Y_M,
    )

    _last_print = 0.0

    try:
        while True:
            frame, depth_frame = cam.get_frames()
            if frame is None:
                continue

            if args.barrels:
                dets = barrel_det.detect(frame, depth_frame)
            else:
                dets = ball_det.detect(frame, depth_frame, target_colour=args.colour)

            now = time.time()
            if now - _last_print > 0.2:
                _last_print = now
                if dets:
                    for d in dets:
                        log.info(
                            "  %-8s  depth=%5.2f m  err=(%+.3f, %+.3f) m  conf=%.0f%%",
                            d.colour,
                            d.depth,
                            d.err_x_m,
                            d.err_y_m,
                            d.conf * 100,
                        )
                else:
                    log.info("  -- no detections --")

            if not args.no_hud:
                vis = frame.copy()
                fc = (cam.width // 2, cam.height // 2)
                cv2.line(
                    vis, (fc[0] - 20, fc[1]), (fc[0] + 20, fc[1]), (255, 255, 255), 1
                )
                cv2.line(
                    vis, (fc[0], fc[1] - 20), (fc[0], fc[1] + 20), (255, 255, 255), 1
                )

                colours_bgr = {
                    "blue": (200, 80, 0),
                    "red": (0, 40, 220),
                    "yellow": (0, 210, 230),
                    "barrel": (150, 150, 0),
                    "unknown": (120, 120, 120),
                }
                for d in dets:
                    col = colours_bgr.get(d.colour, (120, 120, 120))
                    x1, y1, x2, y2 = d.bbox
                    cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
                    cv2.putText(
                        vis,
                        f"{d.colour} {d.depth:.2f}m",
                        (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        col,
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.circle(vis, (d.cx, d.cy), 6, col, -1)
                    # Error arrow (scaled to pixels for visualisation)
                    scale = 200  # px per metre — adjust for screen readability
                    tip_x = fc[0] + int(d.err_x_m * scale)
                    tip_y = fc[1] + int(d.err_y_m * scale)
                    cv2.arrowedLine(vis, fc, (tip_x, tip_y), col, 2, tipLength=0.2)

                info = f"tilt={config.CAMERA_TILT_DEG:.0f}deg  {'BARRELS' if args.barrels else args.colour or 'ALL'}"
                cv2.putText(
                    vis,
                    info,
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow("Camera Test", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        log.info("Stopped.")


if __name__ == "__main__":
    main()
