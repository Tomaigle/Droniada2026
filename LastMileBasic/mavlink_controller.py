import logging
import threading
import time

import config
from pymavlink import mavutil

log = logging.getLogger(__name__)


class MavlinkController:
    def __init__(
        self, port: str = config.MAVLINK_PORT, baud: int = config.MAVLINK_BAUD
    ) -> None:
        self.port = port
        self.baud = baud
        self.mav: mavutil.mavfile | None = None
        self._running = False
        self._hb_thread: threading.Thread | None = None
        self._last_armed = False
        self._last_mode = "UNKNOWN"
        self._last_altitude = 0.0
        self._home_xy = None
        self._last_gps_fix = 0

    def connect(self, timeout: float = config.MAVLINK_CONNECT_TIMEOUT) -> None:
        log.info(f"Connecting to fc on {self.port} @ {self.baud}")
        self.mav = mavutil.mavlink_connection(
            self.port, baud=self.baud, source_system=255, source_component=0
        )
        if not self.mav.wait_heartbeat(timeout=timeout):
            raise ConnectionError(f"No heartbeat received after {timeout}")
        log.info(
            f"FC connected - system {self.mav.target_system}, component {self.mav.target_component}"
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
                0,
                0,
                0,
            )
            time.sleep(1)

    def update_state(self) -> tuple[bool, str]:
        for _ in range(10):
            hb_msg = self.mav.recv_match(type="HEARTBEAT", blocking=False)
            if not hb_msg:
                break
            self._last_mode = mavutil.mode_string_v10(hb_msg)
            self._last_armed = bool(
                hb_msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
            )
        for _ in range(10):
            pos_msg = self.mav.recv_match(type="GLOBAL_POSITION_INT", blocking=False)
            if not pos_msg:
                break
            self._last_altitude = pos_msg.relative_alt / 1000.0
        for _ in range (10):
            gps_msg = self.mav.recv_match(type="GPS_RAW_INT", blocking=False)
            if not gps_msg:
                break
            self._last_gps_fix = gps_msg.fix_type
            if self._last_gps_fix < 5:
                log.warning(f"GPS fix degreded: fix type={self._last_gps_fix}")
        return self._last_armed, self._last_mode

    def is_armed(self) -> bool:
        return self._last_armed

    def get_mode(self) -> str:
        return self._last_mode

    def get_altitude(self) -> float:
        return self._last_altitude

    def send_velocity(self, vx: float, vy: float, vz: float) -> None:
        self.mav.mav.set_position_target_local_ned_send(
            0,
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            0b0000_1111_1100_0111,
            0,
            0,
            0,
            vx,
            vy,
            vz,
            0,
            0,
            0,
            0,
            0,
        )

    def hover(self) -> None:
        self.send_velocity(0.0, 0.0, 0.0)

    def wait_armed(self, timeout=config.MAVLINK_ARM_TIMEOUT) -> bool:
        log.info("Waiting to be armed via RC")
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.update_state()
            if self.is_armed():
                log.info("Drone armed")
                return True
            time.sleep(0.2)
        log.error("Arm timeout")
        return False

    def wait_for_guided(self, timeout=config.MAVLINK_GUIDED_TIMEOUT) -> bool:
        log.info("Waiting for GUIDED mode")
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.update_state()
            if "GUIDED" in self.get_mode().upper():
                return True
            time.sleep(0.5)
        log.error("Timeout waiting for GUIDED")
        return False

    def capture_home(
        self, timeout: float = config.MAVLINK_HOME_CAPTURE_TIMEOUT
    ) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.mav.messages.get("GLOBAL_POSITION_INT")
            if msg and (msg.lat != 0 or msg.lon != 0):
                self._home_xy = (msg.lat, msg.lon)
                log.info(f"Home point saved: {self._home_xy}")
                return True
            time.sleep(0.2)
        log.warning("Home position capture failed")
        return False

    def return_to_launch(self) -> bool:
        mode_id = self.mav.mode_mapping().get("RTL")

        if mode_id is None:
            log.error("RTL mode not found in mode map, attempting fallback return")
            return _fallback_return()

        self.mav.mav.set_mode_send(
            self.mav.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id,
        )

        ack = self.mav.recv_match(type="COMMAND_ACK", blocking=True, timeout=3)
        if ack and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
            log.info("RTL accepted by fc, returning to home")
            return True
        log.error("RTL not acknowledged, attempting fallbacj return")
        return self._fallback_return()

def _fallback_return(self, acceptance_radius = config.MAVLINK_LAND_ERROR_MAX, timeout = )
