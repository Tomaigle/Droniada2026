"""
main.py — Last Mile Logistics BASIC stage entry point.

Operator workflow (SSH from laptop):
  1. Power on drone, props off or safely secured.
  2. ssh pi@<ip>  →  python main.py
  3. Script connects to FC and waits for heartbeat.
  4. Script prints "Arm via RC transmitter..."
  5. Operator arms drone on RC.
  6. Script prints "Type START and press Enter to begin mission."
  7. Operator types START → mission begins.
  8. Ctrl+C at any time → safe abort (gripper open, hover, land).

Flags:
  --sim          Skip MAVLink + gripper hardware (detection only)
  --no-hud       No cv2.imshow (headless SSH session)
  --mock-mav     Real camera, mock MAVLink (print commands only)
  --mock-grip    Real MAVLink, mock gripper
"""

import os
import sys
import time
import argparse
import logging
import threading

os.environ.setdefault("DISPLAY", ":0")
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2

from mavlink_controller import MAVLinkController, MockMAVLinkController
from gripper import Gripper, MockGripper
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


def parse_args():
    p = argparse.ArgumentParser(description="Last Mile Logistics — BASIC stage")
    p.add_argument("--sim", action="store_true", help="No MAVLink, no gripper hardware")
    p.add_argument(
        "--no-hud", action="store_true", help="Disable video window (headless)"
    )
    p.add_argument(
        "--mock-mav", action="store_true", help="Mock MAVLink — print cmds only"
    )
    p.add_argument(
        "--mock-grip", action="store_true", help="Mock gripper — print cmds only"
    )
    return p.parse_args()


# ── Operator start gate ───────────────────────────────────────────────────────


def wait_for_operator_start(event: threading.Event) -> None:
    """Blocks in a thread until operator types START."""
    while True:
        try:
            cmd = input().strip().upper()
        except EOFError:
            break
        if cmd == "START":
            log.info("Operator start command received.")
            event.set()
            break
        else:
            print("Type START to begin mission.")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    # Hardware
    camera = RealSenseCamera()
    camera.start()
    log.info("RealSense ready")

    ball_det = BallDetector(camera)
    barrel_det = BarrelDetector(camera)

    if args.sim or args.mock_mav:
        mav = MockMAVLinkController()
        mav.connect()
    else:
        mav = MAVLinkController()
        mav.connect()

    if args.sim or args.mock_grip:
        gripper = MockGripper()
    else:
        gripper = Gripper()
    gripper.connect()

    # Operator start event
    start_event = threading.Event()
    if args.sim:
        log.info("SIM mode — auto-starting mission")
        start_event.set()
    else:
        log.info("=" * 55)
        log.info("Arm drone via RC transmitter, then type START + Enter")
        log.info("=" * 55)
        t = threading.Thread(
            target=wait_for_operator_start, args=(start_event,), daemon=True
        )
        t.start()

    mission = MissionController(
        mav=mav,
        gripper=gripper,
        camera=camera,
        ball_det=ball_det,
        barrel_det=barrel_det,
        operator_start=start_event,
    )

    # ── Main loop ─────────────────────────────────────────────────────────────
    try:
        while True:
            frame, depth_frame = camera.get_frames()
            if frame is None:
                continue

            telemetry = mission.run_step(frame, depth_frame)

            # Detections for overlay (always run)
            colour = (
                mission.ball_records[mission.current_ball_idx].colour
                if mission.ball_records
                and mission.current_ball_idx < len(mission.ball_records)
                else None
            )
            ball_dets = ball_det.detect(frame, depth_frame, target_colour=colour)
            barrel_dets = barrel_det.detect(frame, depth_frame)
            all_dets = ball_dets + barrel_dets

            if not args.no_hud:
                annotated = draw_frame(
                    frame,
                    state_name=telemetry.get("state", "—"),
                    telemetry=telemetry,
                    detections=all_dets,
                    holding=telemetry.get("holding", False),
                    queue=telemetry.get("queue", []),
                    vx=telemetry.get("vx", 0.0),
                    vy=telemetry.get("vy", 0.0),
                    vz=telemetry.get("vz", 0.0),
                )
                cv2.imshow("Last Mile", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    log.info("Q pressed — aborting")
                    break

            _log_telemetry(telemetry, all_dets)

            from mission import State

            if mission.state in (State.COMPLETE, State.ABORT):
                log.info("Mission ended: %s", mission.state.name)
                time.sleep(3)
                break

    except KeyboardInterrupt:
        log.info("Ctrl+C — emergency abort")
    finally:
        log.info("Shutdown: opening gripper, hover, disconnect")
        gripper.open()
        mav.hover()
        time.sleep(0.5)
        camera.stop()
        cv2.destroyAllWindows()
        mav.disconnect()
        gripper.disconnect()
        log.info("Done.")


# ── Terminal log throttle ─────────────────────────────────────────────────────

_last_log = 0.0


def _log_telemetry(telemetry: dict, detections: list) -> None:
    global _last_log
    now = time.time()
    if now - _last_log < 0.3:
        return
    _last_log = now
    det_str = "  ".join(f"{d.colour}@{d.depth:.2f}m" for d in detections[:4]) or "—"
    log.info(
        "[%s] hold=%s queue=%s  vx=%+.2f vy=%+.2f vz=%+.2f | %s",
        telemetry.get("state", "?"),
        "Y" if telemetry.get("holding") else "N",
        "→".join(telemetry.get("queue", [])) or "done",
        telemetry.get("vx", 0.0),
        telemetry.get("vy", 0.0),
        telemetry.get("vz", 0.0),
        det_str,
    )


if __name__ == "__main__":
    main()
