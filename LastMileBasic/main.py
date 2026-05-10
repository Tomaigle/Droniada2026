import argparse
import logging
import threading
import time

import config
from mavlink_controller import MAVLinkController, MockMAVLinkController
from gripper import Gripper, MockGripper
from detector import RealsenseCamera, ObjectDetector, BallDetector, BarrelDetector
from mission import MissionController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _operator_input_thread(event: threading.Event) -> None:
    while True:
        try:
            cmd = input().strip().lower()
        except EOFError:
            break
        if cmd == "start":
            log.info("Operator: START received")
            event.set()
            break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Use mock MAV + gripper")
    args = parser.parse_args()

    if args.mock:
        log.info("=== MOCK MODE ===")
        mav = MockMAVLinkController()
        gripper = MockGripper()
    else:
        mav = MAVLinkController()
        gripper = Gripper(mav=mav)

    mav.connect()

    camera = RealsenseCamera()
    if not args.mock:
        camera.start()
        log.info("Warming up camera (%d frames)...", config.REALSENSE_WARMUP_FRAMES)
        for _ in range(config.REALSENSE_WARMUP_FRAMES):
            camera.get_frames()

    obj_det = ObjectDetector(camera)
    ball_det = BallDetector(obj_det)
    barrel_det = BarrelDetector(obj_det)

    operator_start = threading.Event()
    t = threading.Thread(
        target=_operator_input_thread, args=(operator_start,), daemon=True
    )
    t.start()
    log.info("Arm drone, then type 'start' + Enter to begin mission.")

    mission = MissionController(
        mav=mav,
        gripper=gripper,
        camera=camera,
        ball_det=ball_det,
        barrel_det=barrel_det,
        operator_start=operator_start,
    )

    try:
        while True:
            if args.mock:
                tel = mission.run_step(None, None)
            else:
                frame, depth_frame = camera.get_frames()
                if frame is None:
                    log.warning("No frame — skipping step")
                    time.sleep(0.05)
                    continue
                tel = mission.run_step(frame, depth_frame)

            log.debug("TEL %s", tel)

            from mission import State

            if mission.state in (State.COMPLETE, State.ABORT):
                log.info("Mission ended in state: %s", mission.state.name)
                break

            time.sleep(0.05)  # ~20 Hz loop

    except KeyboardInterrupt:
        log.warning("KeyboardInterrupt — emergency landing")
        mav.land()
    finally:
        if not args.mock:
            camera.stop()
        mav.disconnect()


if __name__ == "__main__":
    main()
