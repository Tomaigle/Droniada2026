---
description: "Use when implementing code for Droniada Challenge components. Always reference official datasheets and documentation for hardware, APIs, and libraries before coding."
applyTo: "**"
---

# Droniada Challenge 2026 — Documentation & Reference Sources

## Core Documentation

| Document | Location | Purpose |
|----------|----------|---------|
| **Regulamin (Challenge Rules)** | `dokumentacja/Regulamin_konkursu_Droniada_Challenge_2026.pdf` | Official rules, constraints, and project requirements |
| **Firmware** | `copter-MatekF405-TE-72cdb3e15dcc93cc739e57642097bed1/arducopter*` | Flight controller firmware (currently Bcube Pixhawk 6C Pro) |

---

## Component-Specific Documentation & References

### GPS & Telemetry (`maly_dron/`, `LastMileBasic/`)
- **ArduPilot MAVLink Protocol**: https://mavlink.io/
- **Raspberry Pi Documentation**: https://www.raspberrypi.com/documentation/
- **M10 GPS Kit**: Find Septentrio M10 receiver docs (current GPS hardware)
- **v6 Telemetry Radio**: Reference telemetry radio v6 documentation & pinouts
- **ServoHandler & Telemetry**: Reference existing code in `maly_dron/mavlink.py`, `telemetry_log.py`

### Gripper & Delivery (`LastMileBasic/`)
- **Servo Motor Datasheet**: Reference servo model in hardware (typically MG90S or similar)
- **GPIO Control**: https://www.raspberrypi.com/documentation/computers/raspberry-pi.html
- **Existing Implementation**: `LastMileBasic/gripper.py`, `LastMileBasic/mission.py`

### Flight Control & MAVLink (`maly_dron/`, `LastMileBasic/`)
- **MAVLink Commands**: https://mavlink.io/en/messages/common.html
- **ArduCopter Parameters**: https://ardupilot.org/copter/docs/parameters.html
- **Existing Implementation**: `maly_dron/mavlink.py`, `LastMileBasic/mavlink_controller.py`

### ArduCopter Firmware & Flight Controller (`copter-MatekF405-TE-72cdb3e15dcc93cc739e57642097bed1/`)
- **ArduCopter Official Docs**: https://ardupilot.org/copter/
- **ArduCopter Parameters**: https://ardupilot.org/copter/docs/parameters.html
- **Flight Modes**: https://ardupilot.org/copter/docs/flight-modes.html
- **Bcube Pixhawk 6C Pro Specific**: https://bcube.co.uk/ or manufacturer docs (board pinout, I/O capabilities, sensor integrations)
- **Mission Planner**: https://ardupilot.org/planner/docs/mission-planner.html
- **Local Firmware Files**: `copter-MatekF405-TE-72cdb3e15dcc93cc739e57642097bed1/arducopter` (binary), `arducopter.apj` (firmware package)
- **Configuration & Tuning**: https://ardupilot.org/copter/docs/initial-setup.html

### Camera & Vision (`LastMileBasic/`, `ogien_i_woda_basic/`, `realsense/`)
- **OpenCV Documentation**: https://docs.opencv.org/
- **RealSense SDK**: https://github.com/IntelRealSense/librealsense/tree/master/doc
- **RealSense Python Wrapper**: `pip install pyrealsense2`
- **Existing Implementation**: `LastMileBasic/detector.py`, `realsense/` folder

### Audio Classification (`ogien_i_woda_basic/`, `ogien_i_woda_advanced/`)
- **YAMNet (Audio Event Detection)**: https://github.com/tensorflow/models/tree/master/research/audioset/yamnet
- **TensorFlow Audio**: https://www.tensorflow.org/tutorials/audio
- **Existing Implementation**: `ogien_i_woda_advanced/Klasyfikator_Yamnet/ai_engine.py`, `trenuj.py`

### Image Classification & Anomaly Detection (`ogien_i_woda_basic/`, `ogien_i_woda_advanced/`)
- **PyTorch/YOLOv8**: https://docs.ultralytics.com/
- **TensorFlow/Keras**: https://www.tensorflow.org/
- **Image Generation & Augmentation**: `ogien_i_woda_basic/v1/images/generate_panel_images.py`
- **Model Training**: Reference existing scripts in `ogien_i_woda_advanced/`

---

## Before Writing Code: Reference Checklist

✅ **For each component:**
1. **Read the existing code** — How is this component currently used? (`read-before-edit.instructions.md`)
2. **Find the official datasheet** — Search for "[component name] datasheet" or check manufacturer docs
3. **Check the API/Library docs** — Use official references (links above)
4. **Review similar implementations** — Look for working code in the project
5. **Understand safety constraints** — Reference `droniada-challenge-rules.instructions.md`
6. **Plan your approach** — What edge cases exist? What could fail?

---

## Common Hardware References

### Micro-Controllers & Boards
- **Bcube Pixhawk 6C Pro Flight Controller**: https://bcube.co.uk/ (pinout, sensor interfaces, I/O channels)
- **Raspberry Pi**: https://www.raspberrypi.com/documentation/
- **STM32 (flight controller core)**: https://www.st.com/en/microcontrollers/
- **Legacy Reference (Matek F405-TE)**: https://www.mateksys.com/ (if needed for comparison)

### Sensors & Actuators
- **Servo Motors**: Search "[servo model] datasheet" (e.g., MG90S, DS3218)
- **M10 GPS Kit**: Septentrio M10 receiver datasheet and integration guide
- **Telemetry Radio v6**: v6 telemetry radio pinout and protocol documentation
- **Cameras**: Official camera manufacturer docs (Pi Camera, RealSense, etc.)
- **Audio Sensors**: Datasheet from manufacturer (typically I2S MEMS microphone)

### Communication
- **UART/Serial**: Component-specific baud rate & pinout (check firmware/datasheet)
- **I2C/SPI**: Protocol standards (I2C: https://www.i2c-bus.org/)
- **MAVLink**: https://mavlink.io/

---

## Link Verification & Fallback Sources

⚠️ **Some documentation links may be unreliable, outdated, or region-restricted.** Before using a link:

1. **Test the link** — Click it or verify it's accessible
2. **If link is broken**, use **Fallback Sources** below
3. **Document working alternatives** — Update this section with verified sources

### Known Unreliable / Fallback Sources

| Component | Primary Link | Status | Fallback / Alternative |
|-----------|---|---|---|
| **Bcube Pixhawk 6C Pro** | https://bcube.co.uk/ | ⚠️ Unreliable | Search "Pixhawk 6C Pro datasheet" + GitHub ArduPilot board docs |
| **M10 GPS Kit** | Septentrio official | ⛔ May require auth | Google "Septentrio M10 datasheet PDF" or check local project docs |
| **v6 Telemetry Radio** | Manufacturer site | 🔍 Verify | Search part number in project or ask team |

### Fallback Search Strategies

✅ **When a link fails:**
1. Search **GitHub** for `[component] datasheet` or `[component] pinout`
2. Search **ArduPilot forums** or official docs for board-specific info
3. Check **local project files** — datasheets may be stored in `dokumentacja/` or component folders
4. Search **manufacturer archives** — older versions of docs are often available
5. **Ask the team** — someone may have a working alternative or local copy

---

## When You Don't Know a Component

1. **Search the codebase** — Use `grep` or search tools to find existing usage
2. **Look for test files** — `test_*.py` files often show how components work
3. **Check imports & libraries** — The imports tell you what SDK/API is used
4. **Read existing config** — `config.py` files document settings & parameters
5. **Ask before implementing** — If unsure, clarify the component's purpose first (see `droniada-challenge-rules.instructions.md`)

