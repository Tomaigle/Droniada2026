#!/usr/bin/env python3

import csv
import math
import os
import time
from datetime import datetime
from pymavlink import mavutil

PORT = "/dev/ttyACM0"
BAUDRATE = 57600
STREAM_RATE = 10  # Hz
LOG_DIR = os.path.expanduser("./logs")
RC_FAILSAFE_THRESHOLD = 900

# MAV_MODE_FLAG_SAFETY_ARMED = 0x80


def rad2deg(r: float) -> float:
    return math.degrees(r)


def is_armed(base_mode: int) -> bool:
    return True


def has_rc(rc_msg) -> bool:
    return rc_msg is not None and rc_msg.chan3_raw > RC_FAILSAFE_THRESHOLD


def connect() -> mavutil.mavfile:
    print(f"[*] Connecting to {PORT} @ {BAUDRATE} baud ...")
    mav = mavutil.mavlink_connection(PORT, baud=BAUDRATE)
    print("[*] Waiting for heartbeat ...")
    mav.wait_heartbeat(timeout=30)
    print(f"[+] Heartbeat OK  sysid={mav.target_system}\n")
    return mav


def request_streams(mav: mavutil.mavfile) -> None:
    for stream, rate in [
        (mavutil.mavlink.MAV_DATA_STREAM_EXTRA1, STREAM_RATE),  # ATTITUDE
        (mavutil.mavlink.MAV_DATA_STREAM_RC_CHANNELS, STREAM_RATE),
    ]:
        mav.mav.request_data_stream_send(
            mav.target_system, mav.target_component, stream, rate, 1
        )


def make_csv(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = open(path, "w", newline="")
    fields = [
        "timestamp",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
        "rollspeed_dps",
        "pitchspeed_dps",
        "yawspeed_dps",
        "ch1_roll",
        "ch2_pitch",
        "ch3_throttle",
        "ch4_yaw",
        "ch5",
        "ch6",
        "ch7",
        "ch8",
        "armed",
    ]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    return f, writer


def print_status(roll, pitch, yaw, armed, rc_ok, rows):
    armed_str = "ARMED   " if armed else "DISARMED"
    rc_str = "RC:OK " if rc_ok else "RC:--- "
    print(
        f"\r  [{armed_str}] [{rc_str}]  "
        f"Roll:{roll:+7.2f}  Pitch:{pitch:+7.2f}  Yaw:{yaw % 360:6.2f}  "
        f"rows:{rows}",
        end="",
        flush=True,
    )


def wait_for_rc_and_arm(mav: mavutil.mavfile) -> None:
    print("[*] Waiting for RC signal ...")
    rc_msg = None
    while not has_rc(rc_msg):
        rc_msg = mav.recv_match(type="RC_CHANNELS", blocking=True, timeout=2.0)
        if not has_rc(rc_msg):
            print("\r[.] No RC signal ...", end="", flush=True)
    print("\n[+] RC signal detected")

    print("[*] Waiting for ARM (arm your vehicle to begin logging) ...")
    while True:
        hb = mav.recv_match(type="HEARTBEAT", blocking=True, timeout=2.0)
        if hb and is_armed(hb.base_mode):
            break
        print("\r[.] Disarmed ...", end="", flush=True)
    print("\n[+] ARMED — starting log\n")


def main() -> None:
    mav = connect()
    request_streams(mav)
    # wait_for_rc_and_arm(mav)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(LOG_DIR, f"flight_{ts}.csv")
    f, writer = make_csv(csv_path)
    print(f"[+] Logging to: {csv_path}\n")
    print(f"  {'Status':20}  {'Roll':>9}  {'Pitch':>9}  {'Yaw':>8}  rows")
    print("─" * 68)

    attitude = None
    rc = None
    armed = True
    rows = 0
    last_rc_warn = 0.0

    try:
        while True:
            msg = mav.recv_match(
                type=["ATTITUDE", "RC_CHANNELS", "HEARTBEAT"],
                blocking=True,
                timeout=1.0,
            )
            if msg is None:
                continue

            mt = msg.get_type()

            if mt == "HEARTBEAT":
                armed = is_armed(msg.base_mode)
                if not armed:
                    print(f"\n\n[!] DISARMED — stopping log ({rows} rows saved)")
                    break

            elif mt == "ATTITUDE":
                attitude = msg

            elif mt == "RC_CHANNELS":
                rc = msg
                now = time.time()
                if not has_rc(rc) and now - last_rc_warn > 5:
                    print(f"\n[!] RC failsafe / signal lost at row {rows}")
                    last_rc_warn = now

            # Write only when both messages received at least once
            if attitude and rc:
                roll = round(rad2deg(attitude.roll), 3)
                pitch = round(rad2deg(attitude.pitch), 3)
                yaw = round(rad2deg(attitude.yaw) % 360, 3)

                writer.writerow(
                    {
                        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                        "roll_deg": roll,
                        "pitch_deg": pitch,
                        "yaw_deg": yaw,
                        "rollspeed_dps": round(rad2deg(attitude.rollspeed), 3),
                        "pitchspeed_dps": round(rad2deg(attitude.pitchspeed), 3),
                        "yawspeed_dps": round(rad2deg(attitude.yawspeed), 3),
                        "ch1_roll": rc.chan1_raw,
                        "ch2_pitch": rc.chan2_raw,
                        "ch3_throttle": rc.chan3_raw,
                        "ch4_yaw": rc.chan4_raw,
                        "ch5": rc.chan5_raw,
                        "ch6": rc.chan6_raw,
                        "ch7": rc.chan7_raw,
                        "ch8": rc.chan8_raw,
                        "armed": int(armed),
                    }
                )
                rows += 1
                print_status(roll, pitch, yaw, armed, has_rc(rc), rows)

    except KeyboardInterrupt:
        print(f"\n[*] Interrupted — {rows} rows written")
    finally:
        f.close()
        print(f"[+] Saved: {csv_path}")


if __name__ == "__main__":
    main()
