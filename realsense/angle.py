import pyrealsense2 as rs
import numpy as np
import time
import json
import os

CALIBRATION_FILE = "imu_calibration.json"


def get_angles_from_accel(accel, bias=None):
    ax, ay, az = accel.x, accel.y, accel.z
    if bias:
        ax -= bias["ax"]
        ay -= bias["ay"]
        az -= bias["az"]
    pitch = np.degrees(np.arctan2(ay, np.sqrt(ax**2 + az**2)))
    roll = np.degrees(np.arctan2(ax, np.sqrt(ay**2 + az**2)))
    return pitch, roll


def save_calibration(bias):
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(bias, f, indent=2)
    print(f"Calibration saved to {CALIBRATION_FILE}")


def load_calibration():
    if not os.path.exists(CALIBRATION_FILE):
        return None
    with open(CALIBRATION_FILE, "r") as f:
        bias = json.load(f)
    print(
        f"Loaded calibration: ax={bias['ax']:+.4f}  ay={bias['ay']:+.4f}  az={bias['az']:+.4f}"
    )
    return bias


def run_calibration(pipeline):
    """
    Place the camera flat and level, then collect N samples to compute bias.
    At rest, gravity should read: ax=0, ay=9.81, az=0 (RealSense convention).
    """
    print("\n=== CALIBRATION MODE ===")
    print("Place the camera flat and perfectly level.")
    print("Keep it completely still during calibration.")
    input("Press Enter when ready...")

    SAMPLES = 300
    print(f"Collecting {SAMPLES} samples", end="", flush=True)

    readings = []
    while len(readings) < SAMPLES:
        frames = pipeline.wait_for_frames(timeout_ms=5000)
        accel_frame = frames.first_or_default(rs.stream.accel)
        if not accel_frame:
            continue
        d = accel_frame.as_motion_frame().get_motion_data()
        readings.append((d.x, d.y, d.z))
        if len(readings) % 30 == 0:
            print(".", end="", flush=True)

    print(" done.")

    ax_mean = np.mean([r[0] for r in readings])
    ay_mean = np.mean([r[1] for r in readings])
    az_mean = np.mean([r[2] for r in readings])

    # Expected at rest (flat): ax=0, ay=9.81 m/s², az=0
    # Bias = measured - expected
    GRAVITY = 9.81
    bias = {
        "ax": ax_mean - 0.0,
        "ay": ay_mean - GRAVITY,
        "az": az_mean - 0.0,
    }

    print(f"\nRaw means:  ax={ax_mean:+.4f}  ay={ay_mean:+.4f}  az={az_mean:+.4f}")
    print(
        f"Bias:       ax={bias['ax']:+.4f}  ay={bias['ay']:+.4f}  az={bias['az']:+.4f}"
    )

    ax_std = np.std([r[0] for r in readings])
    ay_std = np.std([r[1] for r in readings])
    az_std = np.std([r[2] for r in readings])
    print(f"Noise std:  ax={ax_std:.4f}  ay={ay_std:.4f}  az={az_std:.4f}")

    save_calibration(bias)
    return bias


def check_imu_support():
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError("No RealSense device found!")

    device = devices[0]
    name = device.get_info(rs.camera_info.name)
    print(f"Device found: {name}")

    has_imu = False
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            if profile.stream_type() in (rs.stream.accel, rs.stream.gyro):
                has_imu = True
    return has_imu


def start_pipeline():
    pipeline = rs.pipeline()
    for accel_fps in [250, 200, 100, 63]:
        try:
            config = rs.config()
            config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, accel_fps)
            pipeline.start(config)
            print(f"IMU streaming at {accel_fps} Hz")
            return pipeline
        except RuntimeError:
            continue
    # fallback: let driver pick rate
    config = rs.config()
    config.enable_stream(rs.stream.accel)
    pipeline.start(config)
    print("IMU streaming at default Hz")
    return pipeline


def main():
    if not check_imu_support():
        print("No IMU found. D435i / D455 required.")
        return

    pipeline = start_pipeline()
    bias = load_calibration()

    if bias is None:
        print("\nNo calibration file found.")
        do_cal = input("Run calibration now? (y/n): ").strip().lower()
        if do_cal == "y":
            bias = run_calibration(pipeline)
        else:
            print("Running without calibration — angles may be offset.")
    else:
        redo = input("Recalibrate? (y/n): ").strip().lower()
        if redo == "y":
            bias = run_calibration(pipeline)

    print("\nStreaming angles... Press Ctrl+C to stop.\n")
    try:
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            accel_frame = frames.first_or_default(rs.stream.accel)
            if not accel_frame:
                continue

            accel_data = accel_frame.as_motion_frame().get_motion_data()
            pitch, roll = get_angles_from_accel(accel_data, bias)

            print(
                f"Pitch: {pitch:+7.2f}°  Roll: {roll:+7.2f}°  |  "
                f"ax={accel_data.x:+6.3f} ay={accel_data.y:+6.3f} az={accel_data.z:+6.3f}"
            )

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
