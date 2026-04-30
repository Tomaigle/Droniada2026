"""
gripper.py — Two-servo antagonist gripper controller.

Servos connect directly to RPi via a simple UART servo controller
(e.g. Pololu Maestro, or custom ESP32 running esp_servo.c protocol).

Protocol sent: "SERVO:<channel>:<pwm>\n"
Compatible with the ESP32 sketch already in the project if needed.

MockGripper is available for bench testing without hardware.
"""

import serial
import time
import logging
import config

log = logging.getLogger(__name__)


class Gripper:
    def __init__(self, port: str = config.GRIPPER_PORT, baud: int = config.GRIPPER_BAUD):
        self.port = port
        self.baud = baud
        self._ser: serial.Serial | None = None
        self.is_closed = False

    def connect(self) -> None:
        self._ser = serial.Serial(self.port, self.baud, timeout=1)
        time.sleep(0.5)          # let UART settle
        log.info("Gripper UART connected on %s", self.port)
        self.open()              # always start open

    def disconnect(self) -> None:
        if self._ser and self._ser.is_open:
            self.open()          # safety — always open on shutdown
            self._ser.close()

    def _send(self, channel: int, pwm: int) -> None:
        if not self._ser or not self._ser.is_open:
            log.warning("Gripper UART not open")
            return
        cmd = f"SERVO:{channel}:{pwm}\n"
        self._ser.write(cmd.encode())
        log.debug("Gripper TX: %s", cmd.strip())

    def open(self) -> None:
        """Actively open gripper — both servos push/pull to open."""
        self._send(config.GRIPPER_CHANNEL_A, config.GRIPPER_OPEN_PWM_A)
        self._send(config.GRIPPER_CHANNEL_B, config.GRIPPER_OPEN_PWM_B)
        self.is_closed = False
        log.info("Gripper OPEN")

    def close(self) -> None:
        """Actively close gripper — both servos push/pull to grip."""
        self._send(config.GRIPPER_CHANNEL_A, config.GRIPPER_CLOSE_PWM_A)
        self._send(config.GRIPPER_CHANNEL_B, config.GRIPPER_CLOSE_PWM_B)
        self.is_closed = True
        log.info("Gripper CLOSED")


class MockGripper:
    """Drop-in replacement for bench testing without servo hardware."""
    def __init__(self):
        self.is_closed = False

    def connect(self) -> None:
        log.info("[MOCK GRIPPER] Connected")

    def disconnect(self) -> None:
        log.info("[MOCK GRIPPER] Disconnected")

    def open(self) -> None:
        self.is_closed = False
        log.info("[MOCK GRIPPER] OPEN")

    def close(self) -> None:
        self.is_closed = True
        log.info("[MOCK GRIPPER] CLOSED")
