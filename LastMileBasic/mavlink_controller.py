"""
mavlink_controller.py — ArduPilot interface (Matek / Pixhawk, MAVLink 2).

Key design decisions vs original code:
  - NO self-arming. Drone must be armed via RC transmitter.
    Code only waits for the armed state before proceeding.
  - NO mode forcing on startup. Operator puts drone in GUIDED via RC,
    code detects it and proceeds.
  - Mission start is gated by operator command from laptop (see main.py).
  - Heartbeat keepalive thread runs continuously once connected.
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
        self._running = False
        self._hb_thread: threading.Thread | None = None

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self, timeout: float = 30.0) -> None:
        log.info("Connecting to flight controller on %s @ %d baud...", self.port, self.baud)
        self.mav = mavutil.mavlink_connection(
            self.port, baud=self.baud, source_system=255, source_component=0
        )
        if not self.mav.wait_heartbeat(timeout=timeout):
            raise ConnectionError(
                "No heartbeat received. Check wiring and SERIALx_PROTOCOL=2 on FC."
            )
        log.info(
            "FC connected — system %d component %d",
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

    # ── State queries ─────────────────────────────────────────────────────────

    def is_armed(self) -> bool:
        """Check current armed state from heartbeat."""
        msg = self.mav.recv_match(type="HEARTBEAT", blocking=True, timeout=2)
        if not msg:
            return False
        return bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

    def get_mode(self) -> str:
        """Return current flight mode string, e.g. 'GUIDED'."""
        msg = self.mav.recv_match(type="HEARTBEAT", blocking=True, timeout=2)
        if not msg:
            return "UNKNOWN"
        return mavutil.mode_string_v10(msg)

    def wait_for_armed(self, timeout: float = 120.0) -> bool:
        """
        Block until FC reports armed state.
        Operator arms via RC transmitter — we just wait.
        Returns True if armed, False if timeout.
        """
        log.info("Waiting for RC arm (arm via transmitter)...")
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_armed():
                log.info("FC is ARMED.")
                return True
            time.sleep(0.5)
        log.warning("Arm timeout after %.0f s", timeout)
        return False

    def wait_for_guided(self, timeout: float = 30.0) -> bool:
        """
        Block until FC is in GUIDED mode.
        Operator sets mode via RC or GCS.
        Returns True when GUIDED, False on timeout.
        """
        log.info("Waiting for GUIDED mode (set via RC/GCS)...")
        deadline = time.time() + timeout
        while time.time() < deadline:
            mode = self.get_mode()
            if "GUIDED" in mode.upper():
                log.info("GUIDED mode confirmed.")
                return True
            time.sleep(0.5)
        log.warning("GUIDED mode not set within %.0f s", timeout)
        return False

    def get_altitude_m(self) -> float:
        """Return current relative altitude (AGL) in metres."""
        msg = self.mav.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=1)
        if not msg:
            return 0.0
        return msg.relative_alt / 1000.0   # mm → m

    # ── Velocity control ──────────────────────────────────────────────────────

    def send_velocity(self, vx: float, vy: float, vz: float) -> None:
        """
        Body-frame velocity command (NED convention).
          vx: forward+  [m/s]
          vy: right+    [m/s]
          vz: down+     [m/s]  (positive = descend)
        """
        self.mav.mav.set_position_target_local_ned_send(
            0,
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            0b0000_1111_1100_0111,          # velocity only
            0, 0, 0,                        # position ignored
            vx, vy, vz,
            0, 0, 0,                        # accel ignored
            0, 0,                           # yaw ignored
        )

    def hover(self) -> None:
        self.send_velocity(0.0, 0.0, 0.0)

    def land(self) -> None:
        """Switch to LAND mode — pilot can override via RC at any time."""
        mode_id = self.mav.mode_mapping().get("LAND")
        if mode_id is None:
            log.error("LAND mode not found in mode map")
            return
        self.mav.mav.set_mode_send(
            self.mav.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id,
        )
        log.info("LAND mode commanded")


class MockMAVLinkController:
    """
    Drop-in replacement for MAVLinkController used in --sim / test modes.
    Prints commands to terminal instead of sending MAVLink packets.
    """
    def connect(self, **_) -> None:
        log.info("[MOCK MAV] Connected (simulation)")

    def disconnect(self) -> None:
        log.info("[MOCK MAV] Disconnected")

    def is_armed(self) -> bool:
        return True

    def get_mode(self) -> str:
        return "GUIDED"

    def wait_for_armed(self, **_) -> bool:
        log.info("[MOCK MAV] Pretending armed")
        return True

    def wait_for_guided(self, **_) -> bool:
        log.info("[MOCK MAV] Pretending GUIDED")
        return True

    def get_altitude_m(self) -> float:
        return config.PICKUP_HOVER_M

    def send_velocity(self, vx: float, vy: float, vz: float) -> None:
        log.debug("[MOCK MAV] VEL vx=%+.2f vy=%+.2f vz=%+.2f", vx, vy, vz)

    def hover(self) -> None:
        log.debug("[MOCK MAV] HOVER")

    def land(self) -> None:
        log.info("[MOCK MAV] LAND")
