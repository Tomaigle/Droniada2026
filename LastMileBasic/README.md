# Last Mile Logistics — BASIC Stage

Autonomous drone: pick up 3 coloured tennis balls (blue → red → yellow),
drop each into a separate blue barrel. Pure vision — no pre-loaded GPS.

## File structure

```
lastmile/
├── config.py             ← ALL tunable parameters
├── main.py               ← Mission entry point
├── mavlink_controller.py ← ArduPilot/MAVLink interface (+ MockMAV)
├── gripper.py            ← Two-servo antagonist gripper over UART (+ MockGripper)
├── detector.py           ← RealSense + YOLO balls + HSV barrels + tilt compensation
├── mission.py            ← State machine
├── overlay.py            ← HUD drawing
├── test_camera.py        ← Camera + detection test (no FC/gripper needed)
├── test_mavlink.py       ← MAVLink comms test + keyboard joystick
├── test_gripper.py       ← Servo open/close test (no FC/camera needed)
└── requirements.txt
```

## Hardware

| Component          | Detail                                          |
|--------------------|-------------------------------------------------|
| Flight controller  | Matek H743 / Pixhawk — ArduPilot, MAVLink 2     |
| Companion computer | Raspberry Pi 5 (16 GB) + AI HAT (13 TOPS)       |
| Camera             | Intel RealSense D4xx (bolted, fixed tilt angle) |
| Gripper            | Two-servo antagonist, RPi UART servo controller |

## ArduPilot FC setup

```
SERIALx_PROTOCOL = 2      # MAVLink 2 on UART to RPi
SERIALx_BAUD     = 115    # 115200
```

## Install

```bash
pip install -r requirements.txt --break-system-packages
```

## Running

### Full mission (SSH from laptop)

```bash
python main.py
```

1. Script connects to FC, waits for heartbeat
2. "Arm via RC transmitter" printed
3. Arm on RC
4. "Type START + Enter" printed
5. Type `START` → mission begins
6. Ctrl+C at any time → safe abort

### Headless (no display over SSH)

```bash
python main.py --no-hud
```

### Full simulation (no hardware at all)

```bash
python main.py --sim
```

### Mixed modes

```bash
python main.py --mock-mav           # real camera + gripper, MAVLink printed only
python main.py --mock-grip          # real camera + FC, gripper printed only
python main.py --mock-mav --no-hud  # fully headless sim
```

## Test tools

```bash
# Camera + detection only (no FC or gripper)
python test_camera.py
python test_camera.py --colour blue    # single colour filter
python test_camera.py --barrels        # barrel detection mode
python test_camera.py --no-hud         # terminal only

# MAVLink only (no camera)
python test_mavlink.py --read-only     # connect + print FC state
python test_mavlink.py --mock          # fake FC, print commands
python test_mavlink.py --joystick      # WASD keyboard → real velocity commands

# Gripper only (no camera or FC)
python test_gripper.py
python test_gripper.py --mock
python test_gripper.py --cycle 10      # stress test 10 open/close cycles
```

## Camera tilt calibration

1. Build drone, bolt RealSense at desired angle.
2. Measure angle with phone inclinometer app.
3. Set `CAMERA_TILT_DEG` in `config.py`.
4. Run `test_camera.py`, place ball directly below drone.
5. Check `err_x_m` and `err_y_m` → should be near zero when centred.
6. Adjust `GRIPPER_OFFSET_X_M` / `GRIPPER_OFFSET_Y_M` until errors centre correctly.

## Mission state machine

```
WAIT_ARM → WAIT_START → REVERSE_TO_BALLS → SCAN_BALLS
  → ALIGN_BALL → GRIP → VERIFY_GRIP ──┐
       ↑ retry if failed ←────────────┘
       ↓ success
  → (next ball or FLY_TO_BARRELS)
  → SCAN_BARRELS → ALIGN_BARREL → DROP
       └─ repeat for each ball ─┘
  → COMPLETE → LAND
```

## Switching to custom model

1. Set `MODEL_PATH = "your_model.pt"` in `config.py`
2. Set `USE_STOCK_MODEL = False`
3. Set `BALL_CLASSES` to your trained class indices

## TODO

- Search pattern is currently a simple forward creep.
  Upgrade to a lawnmower/spiral pattern for reliability in open fields.
- Barrel detector uses HSV colour segmentation.
  Replace with a trained YOLO class for better lighting robustness.
- `used_barrels` tracks by pixel position — add depth-based 3D position
  tracking for robustness across large barrel separations.
- No GPS geofencing yet — add MAVLink fence upload before competition.
- Read thru rules and adjust hover and drop height agl
- Landing spot return not wherever
- figure out online telemetry
- start sequence position adjust
- camera angle calibration
- camera tilt compestation for both x and y
