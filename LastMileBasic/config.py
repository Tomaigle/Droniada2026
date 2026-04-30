"""
config.py — Central configuration for Last Mile Logistics drone mission.
All tunable parameters live here. Edit this file, not the modules.
"""

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_PATH = "new_best.pt"          # swap to custom model when ready
CONF_THRESHOLD = 0.30               # detection confidence minimum

# Ball class IDs in your model.
# When using stock YOLOv8: class 32 = sports ball (no colour separation).
# When using custom model: set these to your trained class indices.
BALL_CLASSES = {
    "blue":   0,
    "red":    1,
    "yellow": 2,
}
# Pickup order required by rulebook: blue → red → yellow
PICKUP_ORDER = ["blue", "red", "yellow"]

# Stock YOLOv8 fallback (single class, no colour separation)
STOCK_BALL_CLASS_ID = 32
USE_STOCK_MODEL = True              # set False when custom model is ready

# ── Camera ────────────────────────────────────────────────────────────────────
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
FPS          = 30

# ── MAVLink / UART ────────────────────────────────────────────────────────────
MAVLINK_PORT      = "/dev/serial0"  # RPi5 primary UART
MAVLINK_BAUD      = 115200
MAVLINK_SYS_ID    = 1               # ArduPilot default

# ── Servo (gripper) ───────────────────────────────────────────────────────────
SERVO_CHANNEL     = 9               # SERVO9 / S5 on Matek/Pixhawk aux
SERVO_OPEN_PWM    = 1000            # µs — gripper open
SERVO_CLOSED_PWM  = 2000            # µs — gripper closed
GRIP_HOLD_S       = 2.0             # seconds to hold before release

# ── Flight control ────────────────────────────────────────────────────────────
# Proportional gain applied to pixel/metric error → velocity command
KP_XY        = 0.5
KP_Z         = 0.4
MAX_SPEED_XY = 0.4                  # m/s horizontal
MAX_SPEED_Z  = 0.3                  # m/s vertical (descent)

# Alignment tolerance before drone is allowed to descend
CENTER_THRESH_M  = 0.12             # m — XY error must be below this

# Heights
PICKUP_HOVER_M   = 1.5              # m — hover height above ball zone
DROP_HOVER_M     = 2.0              # m — hover height above barrel zone
GRIP_DISTANCE_M  = 0.25             # m — depth at which gripper triggers

# Barrel detection
BARREL_CONF      = 0.25             # confidence for barrel detection
BARREL_DIAMETER_M = 0.58            # real barrel opening diameter (m)

# ── Mission timeouts ──────────────────────────────────────────────────────────
MISSION_TIMEOUT_S   = 600           # 10 minutes total
SEARCH_TIMEOUT_S    = 15            # seconds to search before giving up
ALIGN_TIMEOUT_S     = 10            # seconds to align before aborting ball
