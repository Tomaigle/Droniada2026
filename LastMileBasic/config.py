"""
config.py — Central configuration for Last Mile Logistics drone mission.
All tunable parameters live here. Edit this file, not the modules.
"""

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_PATH = "new_best.pt"
CONF_THRESHOLD = 0.30

# Ball class IDs — update when custom model is ready
BALL_CLASSES = {
    "blue": 0,
    "red": 1,
    "yellow": 2,
}
PICKUP_ORDER = ["blue", "red", "yellow"]  # rulebook: N → C → Ż
STOCK_BALL_CLASS_ID = 32
USE_STOCK_MODEL = True  # set False when custom model ready

# ── Camera ────────────────────────────────────────────────────────────────────
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# Physical camera tilt downward from horizontal (degrees).
# Measure with phone inclinometer once drone is built.
CAMERA_TILT_DEG = 30.0

# Offset from camera optical axis to gripper tip, in camera frame (metres).
# Positive Y = gripper is ahead of camera along the tilt axis.
# Measure physically once mounted.
GRIPPER_OFFSET_X_M = 0.00
GRIPPER_OFFSET_Y_M = 0.15

# ── MAVLink (flight controller) ───────────────────────────────────────────────
MAVLINK_PORT = "/dev/serial0"  # RPi5 UART to FC
MAVLINK_BAUD = 115200
MAVLINK_SYS_ID = 1

# ── Gripper (two-servo antagonist, direct RPi UART) ───────────────────────────
# Servo A and B work like bicep/tricep — one pulls, other pushes.
GRIPPER_PORT = "/dev/ttyAMA2"  # separate UART for servo controller
GRIPPER_BAUD = 115200

GRIPPER_CHANNEL_A = 1
GRIPPER_CHANNEL_B = 2

GRIPPER_CLOSE_PWM_A = 2000  # µs — A closes
GRIPPER_CLOSE_PWM_B = 1000  # µs — B closes
GRIPPER_OPEN_PWM_A = 1000  # µs — A opens
GRIPPER_OPEN_PWM_B = 2000  # µs — B opens

GRIP_HOLD_S = 1.5  # s — hold before verifying
GRIP_VERIFY_WAIT_S = 0.4  # s — pause for servo to settle before check

# ── Grip verification ─────────────────────────────────────────────────────────
# After gripping, hover and re-run detector on the target ball colour.
# If ball still visible at similar depth → grip failed → retry.
GRIP_VERIFY_ENABLED = True
GRIP_VERIFY_DEPTH_DELTA = 0.10  # m — depth must drop by at least this on success
MAX_GRIP_RETRIES = 3  # attempts before giving up on a ball

# ── Flight control ────────────────────────────────────────────────────────────
KP_XY = 0.5
KP_Z = 0.4
MAX_SPEED_XY = 0.4  # m/s horizontal
MAX_SPEED_Z = 0.3  # m/s vertical

CENTER_THRESH_M = 0.10  # m — XY error threshold to allow descent
GRIP_DISTANCE_M = 0.20  # m — depth at which gripper fires

# ── Mission geometry ──────────────────────────────────────────────────────────
PICKUP_HOVER_M = 1.5  # m AGL during transit / search
DROP_HOVER_M = 2.0  # m AGL above barrel for drop

# Balls are ~5 m behind start. Drone reverses this far to reach them.
INITIAL_REVERSE_M = 5.5  # m
INITIAL_REVERSE_SPEED = 0.3  # m/s

# After collecting balls, fly forward to barrel zone.
BARREL_SEARCH_FWD_M = 8.0  # m (approx — barrels acquired by vision)
BARREL_SEARCH_SPEED = 0.3  # m/s

# ── Barrel detection ──────────────────────────────────────────────────────────
BARREL_CONF = 0.25
BARREL_DIAMETER_M = 0.58

# ── Mission timeouts ──────────────────────────────────────────────────────────
MISSION_TIMEOUT_S = 600  # 10 min hard cutoff
ALIGN_TIMEOUT_S = 15  # s per alignment attempt
SEARCH_TIMEOUT_S = 20  # s before expanding search
