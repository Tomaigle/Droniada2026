"""
mavlink_controller.py — MAVLink interface for ArduPilot (Matek/Pixhawk).

Handles:
  - Connection + heartbeat keepalive thread
  - Guided mode arming / takeoff
  - SET_POSITION_TARGET_LOCAL_NED velocity commands
  - Servo (gripper) control
  - Land command
"""

import time
import threading
import logging
from pymavlink import mavutil
import config

log = logging.getLogger(__name__)


class MAVLinkController:
    def __init__(self, port: str = config.MAVLINK_PORT, baud: int = config.MAVLINK_BAUD):
        self.port = port
        self.baud = baud
        self.mav: mavutil.mavfile | None = None
        self._hb_thread: threading.Thread | None = None
        self._running = False

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self, timeout: float = 15.0) -> None:
        log.info("Connecting to ArduPilot on %s @ %d baud...", self.port, self.baud)
        self.mav = mavutil.mavlink_connection(
            self.port, baud=self.baud, source_system=255, source_component=0
        )
        if not self.mav.wait_heartbeat(timeout=timeout):
            raise ConnectionError("No heartbeat received — check wiring and SERIALx_PROTOCOL=2")
        log.info(
            "Heartbeat OK — system %d component %d",
            self.mav.target_system,
            self.mav.target_component,
        )
        self._running = True
        self._hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._hb_thread.start()

    def disconnect(self) -> None:
        self._running = False
        if self.mav:
            self.mav.close()

    def _heartbeat_loop(self) -> None:
        while self._running:
            self.mav.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_GCS,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0, 0, 0,
            )
            time.sleep(1)

    # ── Mode / arming ─────────────────────────────────────────────────────────

    def set_mode(self, mode: str) -> None:
        """Set flight mode by name, e.g. 'GUIDED', 'LAND'."""
        mode_id = self.mav.mode_mapping().get(mode)
        if mode_id is None:
            raise ValueError(f"Unknown mode: {mode}")
        self.mav.mav.set_mode_send(
            self.mav.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id,
        )
        log.info("Mode set → %s", mode)

    def arm(self, force: bool = False) -> None:
        param2 = 21196 if force else 0
        self.mav.mav.command_long_send(
            self.mav.target_system, self.mav.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, param2, 0, 0, 0, 0, 0,
        )
        log.info("Arm command sent")

    def takeoff(self, altitude_m: float) -> None:
        self.mav.mav.command_long_send(
            self.mav.target_system, self.mav.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, altitude_m,
        )
        log.info("Takeoff to %.1f m", altitude_m)

    def land(self) -> None:
        self.set_mode("LAND")
        log.info("LAND mode set")

    # ── Velocity control ──────────────────────────────────────────────────────

    def send_velocity(self, vx: float, vy: float, vz: float) -> None:
        """
        Send body-frame velocity command.
        vx: forward+ / back-   [m/s]
        vy: right+  / left-    [m/s]
        vz: down+   / up-      [m/s]  (NED convention — positive = descend)
        """
        self.mav.mav.set_position_target_local_ned_send(
            0,                                          # time_boot_ms (ignored)
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            0b0000_1111_1100_0111,                      # type_mask: velocity only
            0, 0, 0,                                    # position (ignored)
            vx, vy, vz,                                 # velocity
            0, 0, 0,                                    # acceleration (ignored)
            0, 0,                                       # yaw, yaw_rate (ignored)
        )

    def hover(self) -> None:
        """Stop all motion — send zero velocity."""
        self.send_velocity(0.0, 0.0, 0.0)

    # ── Servo (gripper) ───────────────────────────────────────────────────────

    def set_servo(self, channel: int, pwm: int) -> None:
        self.mav.mav.command_long_send(
            self.mav.target_system, self.mav.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
            0,
            float(channel),
            float(pwm),
            0, 0, 0, 0, 0,
        )

    def gripper_open(self) -> None:
        self.set_servo(config.SERVO_CHANNEL, config.SERVO_OPEN_PWM)
        log.info("Gripper OPEN")

    def gripper_close(self) -> None:
        self.set_servo(config.SERVO_CHANNEL, config.SERVO_CLOSED_PWM)
        log.info("Gripper CLOSED")
