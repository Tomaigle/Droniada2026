# Last Mile Logistics — BASIC Stage

Autonomous drone software: pick up 3 coloured tennis balls (blue → red → yellow)
and drop each into a separate blue barrel. Purely vision-based; no pre-loaded GPS coords.

## File structure

```
lastmile/
├── config.py             ← ALL tunable parameters (edit here first)
├── main.py               ← Entry point
├── mavlink_controller.py ← ArduPilot/MAVLink interface
├── detector.py           ← RealSense camera + YOLO ball + HSV barrel detection
├── mission.py            ← State machine (SEARCH → ALIGN → GRIP → DROP)
├── overlay.py            ← OpenCV HUD drawing
└── requirements.txt
```

## Hardware assumptions

| Component         | Detail                                      |
|-------------------|---------------------------------------------|
| Flight controller | Matek H743 / Pixhawk — ArduPilot, MAVLink 2 |
| Companion computer| Raspberry Pi 5 (16 GB) + AI HAT (13 TOPS)  |
| Camera            | Intel RealSense D4xx (colour + depth)       |
| Gripper           | Servo on SERVO9 / S5 aux output             |
| UART              | RPi5 `/dev/serial0` ↔ FC UART               |

## ArduPilot setup

```
SERIALx_PROTOCOL = 2      # MAVLink 2 on the UART connected to RPi
SERIALx_BAUD     = 115    # 115200
SERVO9_FUNCTION  = 0      # passthrough / manual
```

## Install

```bash
pip install -r requirements.txt --break-system-packages
```

## Run

```bash
# Full autonomous mission
python main.py

# Detection-only debug (no FC needed, no arming)
python main.py --sim

# Headless (no display)
python main.py --no-hud
```

## Switching to custom model

When your coloured-ball model is ready:

1. Set `MODEL_PATH = "your_model.pt"` in `config.py`
2. Set `USE_STOCK_MODEL = False`
3. Update `BALL_CLASSES` dict with your class indices

Everything else adapts automatically.

## State machine

```
IDLE → TAKEOFF → SEARCH_BALL → ALIGN_BALL → GRIP
                                               ↓
              SEARCH_BARREL ← ← ← ← ← ← ← ← ─┘
                   ↓
              ALIGN_BARREL → DROP ──┐
                                    ├─ more balls? → SEARCH_BALL
                                    └─ done?       → COMPLETE → LAND
```

## Known limitations / TODO

- Search pattern is currently a simple forward creep.
  Upgrade to a lawnmower/spiral pattern for reliability in open fields.
- Barrel detector uses HSV colour segmentation.
  Replace with a trained YOLO class for better lighting robustness.
- `used_barrels` tracks by pixel position — add depth-based 3D position
  tracking for robustness across large barrel separations.
- No GPS geofencing yet — add MAVLink fence upload before competition.
