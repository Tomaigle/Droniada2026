"""
main.py — Last Mile Logistics BASIC stage entry point.

Usage:
    python main.py            # full autonomous mission
    python main.py --sim      # skip MAVLink/camera, run detector only (debug)
    python main.py --no-hud   # headless (no cv2.imshow)

Ctrl+C or press Q in the HUD window to abort safely.
"""

import os
import sys
import time
import argparse
import logging

os.environ.setdefault("DISPLAY", ":0")
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2

from mavlink_controller import MAVLinkController
from detector import RealSenseCamera, BallDetector, BarrelDetector
from mission import MissionController
from overlay import draw_frame
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Last Mile Logistics — BASIC stage")
    p.add_argument("--sim",    action="store_true", help="Skip MAVLink; detect-only mode")
    p.add_argument("--no-hud", action="store_true", help="Disable live video window")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Hardware init ──────────────────────────────────────────────────────
    camera = RealSenseCamera()
    camera.start()
    log.info("Camera ready")

    ball_det   = BallDetector(camera)
    barrel_det = BarrelDetector(camera)

    mav = MAVLinkController()
    if not args.sim:
        mav.connect()
        mav.gripper_open()          # ensure gripper starts open
        log.info("MAVLink ready")
    else:
        log.info("SIM mode — MAVLink disabled")

    mission = MissionController(mav, camera, ball_det, barrel_det)

    if not args.sim:
        input("Press ENTER to arm and start mission...")
        mission.start()
    else:
        log.info("SIM: detection-only loop running. Press Q to quit.")

    # ── Main loop ──────────────────────────────────────────────────────────
    try:
        while True:
            frame, depth_frame = camera.get_frames()
            if frame is None:
                continue

            # Detection (always run for overlay, even in sim)
            current_colour = mission.current_colour if not args.sim else None
            ball_dets   = ball_det.detect(frame, depth_frame, target_colour=current_colour)
            barrel_dets = barrel_det.detect(frame, depth_frame)
            all_dets    = ball_dets + barrel_dets

            # Mission step
            telemetry = mission.run_step(frame, depth_frame) if not args.sim else {}

            # HUD
            if not args.no_hud:
                annotated = draw_frame(
                    frame,
                    state_name=telemetry.get("state", "SIM"),
                    telemetry=telemetry,
                    detections=all_dets,
                    holding=telemetry.get("holding", False),
                    queue=telemetry.get("queue", list(config.PICKUP_ORDER)),
                    vx=telemetry.get("vx", 0.0),
                    vy=telemetry.get("vy", 0.0),
                    vz=telemetry.get("vz", 0.0),
                )
                cv2.imshow("Last Mile — Ball Tracker", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    log.info("Q pressed — aborting")
                    break

            # Terminal log (throttled)
            _log_telemetry(telemetry, all_dets)

            # Mission complete check
            from mission import State
            if mission.state in (State.COMPLETE, State.ABORT):
                log.info("Mission ended in state %s. Exiting loop.", mission.state.name)
                time.sleep(3)
                break

    except KeyboardInterrupt:
        log.info("Keyboard interrupt")
    finally:
        log.info("Shutting down...")
        if not args.sim:
            mav.gripper_open()
            mav.hover()
        camera.stop()
        cv2.destroyAllWindows()
        log.info("Done.")


# ── Terminal telemetry ─────────────────────────────────────────────────────────
_last_log = 0.0

def _log_telemetry(telemetry: dict, detections: list) -> None:
    global _last_log
    now = time.time()
    if now - _last_log < 0.25:
        return
    _last_log = now

    state = telemetry.get("state", "-")
    queue = telemetry.get("queue", [])
    hold  = telemetry.get("holding", False)
    vx    = telemetry.get("vx", 0.0)
    vy    = telemetry.get("vy", 0.0)
    vz    = telemetry.get("vz", 0.0)

    det_str = "  ".join(
        f"{d.colour}@{d.depth:.2f}m" for d in detections[:3]
    ) or "no detections"

    log.info(
        "[%s] grip=%s queue=%s  vx=%+.2f vy=%+.2f vz=%+.2f  | %s",
        state, "HOLD" if hold else "open",
        "→".join(queue) if queue else "done",
        vx, vy, vz, det_str,
    )


if __name__ == "__main__":
    main()
