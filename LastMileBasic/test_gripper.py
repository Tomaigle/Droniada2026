"""
test_gripper.py — Gripper servo test. No FC or camera needed.

Commands:
  python test_gripper.py           # interactive open/close via keyboard
  python test_gripper.py --mock    # mock gripper, no hardware
  python test_gripper.py --cycle N # auto-cycle N times (stress test)
"""

import time, argparse, logging
from gripper import Gripper, MockGripper
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Gripper test")
    p.add_argument("--mock",  action="store_true")
    p.add_argument("--cycle", type=int, default=0, help="Auto-cycle N times")
    return p.parse_args()


def main():
    args = parse_args()
    g = MockGripper() if args.mock else Gripper()
    g.connect()

    if args.cycle:
        log.info("Cycling gripper %d times (open → close → open, 1 s each)", args.cycle)
        for i in range(args.cycle):
            log.info("Cycle %d/%d — CLOSE", i+1, args.cycle)
            g.close()
            time.sleep(1.0)
            log.info("Cycle %d/%d — OPEN", i+1, args.cycle)
            g.open()
            time.sleep(1.0)
        log.info("Done.")
        g.disconnect()
        return

    log.info("Interactive gripper test")
    print("Commands: o=open  c=close  q=quit")
    while True:
        cmd = input("> ").strip().lower()
        if cmd == "q":
            break
        elif cmd == "o":
            g.open()
        elif cmd == "c":
            g.close()
        else:
            print("Unknown: o / c / q")

    g.disconnect()


if __name__ == "__main__":
    main()
