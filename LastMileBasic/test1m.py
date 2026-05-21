import time
import logging
import config
from mavlink_controller import MAVLinkController

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

HOVER_ALT_M = 1.0
HOVER_DURATION_S = 5.0
ARM_POLL_INTERVAL = 1.0
ARM_TIMEOUT_S = 60.0
TAKEOFF_SETTLE_S = 5.0
LAND_TIMEOUT_S = 30.0
FORWARD_SPEED_MS = 0.3
FORWARD_DIST_M = 1.0
FORWARD_DURATION_S = FORWARD_DIST_M / FORWARD_SPEED_MS


def wait_armed(mav: MAVLinkController, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if mav.is_armed():
            return True
        time.sleep(ARM_POLL_INTERVAL)
    return False


def wait_mode(mav: MAVLinkController, mode: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if mav.get_state_snapshot()["mode"].upper() == mode.upper():
            return True
        time.sleep(0.5)
    return False


def main():
    mav = MAVLinkController()
    mav.connect(timeout=config.MAVLINK_CONNECT_TIMEOUT)

    log.info("Waiting for arm...")
    if not wait_armed(mav, ARM_TIMEOUT_S):
        log.error("Drone not armed within %ss — abort", ARM_TIMEOUT_S)
        mav.disconnect()
        return

    log.info("Setting GUIDED mode")
    mav.set_mode("GUIDED")
    if not wait_mode(mav, "GUIDED", config.MAVLINK_GUIDED_TIMEOUT):
        log.error("GUIDED mode not confirmed — abort")
        mav.disconnect()
        return

    log.info("Takeoff to %.1f m", HOVER_ALT_M)
    mav.takeoff(HOVER_ALT_M)
    time.sleep(TAKEOFF_SETTLE_S)

    log.info("Hovering for %.1f s", HOVER_DURATION_S)
    mav.hover()
    time.sleep(HOVER_DURATION_S)

    log.info(
        "Moving forward 1 m at %.1f m/s for %.1f s",
        FORWARD_SPEED_MS,
        FORWARD_DURATION_S,
    )
    mav.send_velocity(FORWARD_SPEED_MS, 0.0, 0.0)
    time.sleep(FORWARD_DURATION_S)
    mav.hover()
    time.sleep(1.0)

    log.info("Landing")
    mav.land()

    deadline = time.monotonic() + LAND_TIMEOUT_S
    while time.monotonic() < deadline:
        state = mav.get_state_snapshot()
        if state["mode"].upper() == "LAND":
            log.info("LAND mode confirmed, alt=%.2f m", state["alt_m"])
        if state["alt_m"] < 0.1:
            log.info("Landed")
            break
        time.sleep(1.0)

    mav.disconnect()


if __name__ == "__main__":
    main()
