"""
test_mavlink.py — MAVLink command printer.
Connects to real FC, reads arm/mode state, sends velocity commands
to terminal only (MockMAV) or to real FC in dry-run mode.

Use this to verify:
  - UART connection to FC works
  - Heartbeat is received
  - Armed / mode state reads correctly
  - Velocity commands are formed correctly

Modes:
  python test_mavlink.py --read-only   # connect + print FC state, no commands sent
  python test_mavlink.py --mock        # fake FC, print commands only
  python test_mavlink.py --joystick    # WASD keyboard → velocity commands to real FC
                                       # (good for testing FC accepts commands)

WARNING: --joystick sends real MAVLink commands. Drone must be on ground,
         props off, or safely restrained.
"""

import os, sys, time, argparse, logging, threading
import termios, tty

from mavlink_controller import MAVLinkController, MockMAVLinkController
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="MAVLink test tool")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--read-only", action="store_true", help="Connect, read state, no commands")
    g.add_argument("--mock",      action="store_true", help="Use mock MAV (no hardware)")
    g.add_argument("--joystick",  action="store_true", help="WASD keyboard velocity test")
    return p.parse_args()


# ── Read-only test ────────────────────────────────────────────────────────────

def run_read_only(mav: MAVLinkController):
    log.info("Read-only mode — polling FC state every 1 s. Ctrl+C to stop.")
    while True:
        armed  = mav.is_armed()
        mode   = mav.get_mode()
        alt    = mav.get_altitude_m()
        log.info("  armed=%-5s  mode=%-10s  alt=%.2f m", armed, mode, alt)
        time.sleep(1.0)


# ── Keyboard joystick ─────────────────────────────────────────────────────────

_SPEED = 0.2   # m/s for keyboard test

def _getch():
    """Read single keypress without Enter."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def run_joystick(mav: MAVLinkController):
    log.info("Joystick mode — WASD = forward/back/left/right, RF = up/down, Q = quit")
    log.info("Drone must be in GUIDED mode. Commands sent continuously while key held.")
    log.info("Press any key once to start...")
    _getch()

    vx = vy = vz = 0.0
    last_cmd = time.time()

    print("\nControls: W=fwd S=back A=left D=right R=up F=down SPACE=hover Q=quit\n")

    while True:
        ch = _getch().lower()
        now = time.time()

        if ch == "q":
            mav.hover()
            log.info("Quit — hover commanded")
            break
        elif ch == "w":   vx, vy, vz = +_SPEED, 0, 0
        elif ch == "s":   vx, vy, vz = -_SPEED, 0, 0
        elif ch == "a":   vx, vy, vz = 0, -_SPEED, 0
        elif ch == "d":   vx, vy, vz = 0, +_SPEED, 0
        elif ch == "r":   vx, vy, vz = 0, 0, -_SPEED   # up = negative NED Z
        elif ch == "f":   vx, vy, vz = 0, 0, +_SPEED
        elif ch == " ":   vx, vy, vz = 0, 0, 0

        mav.send_velocity(vx, vy, vz)
        log.info("  VEL  vx=%+.2f  vy=%+.2f  vz=%+.2f  m/s", vx, vy, vz)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.mock:
        mav = MockMAVLinkController()
        mav.connect()
        log.info("Mock MAV connected — printing commands to terminal")
        log.info("Sending test velocity sequence...")
        for vx, vy, vz, label in [
            (+0.3,  0.0,  0.0, "forward"),
            ( 0.0, +0.3,  0.0, "right"),
            ( 0.0,  0.0, +0.2, "descend"),
            ( 0.0,  0.0,  0.0, "hover"),
        ]:
            log.info("  %-10s  vx=%+.2f vy=%+.2f vz=%+.2f", label, vx, vy, vz)
            mav.send_velocity(vx, vy, vz)
            time.sleep(1.0)
        return

    mav = MAVLinkController()
    try:
        mav.connect()
    except ConnectionError as e:
        log.error("Connection failed: %s", e)
        sys.exit(1)

    try:
        if args.joystick:
            run_joystick(mav)
        else:
            run_read_only(mav)
    except KeyboardInterrupt:
        log.info("Interrupted")
    finally:
        mav.hover()
        mav.disconnect()


if __name__ == "__main__":
    main()
