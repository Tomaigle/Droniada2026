import logging
import math
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
        self._last_lat = 0
        self._last_lon = 0
        self._last_heading = 0.0
        self._last_gps_fix = 0
        self._home_xy = None
        self._geofence_radial: float | None = None
        self._geofence_square: tuple | None = None

    def connect(self, timeout: float = config.MAVLINK_CONNECT_TIMEOUT) -> None:
        log.info(f"Connecting to FC on {self.port} @ {self.baud}")
        self.mav = mavutil.mavlink_connection(
            self.port, baud=self.baud, source_system=255, source_component=0
        )
        if not self.mav.wait_heartbeat(timeout=timeout):
            raise ConnectionError(f"No heartbeat received after {timeout}s")
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
            self._last_lat = pos_msg.lat
            self._last_lon = pos_msg.lon
            self._last_heading = pos_msg.hdg / 100.0
        for _ in range(10):
            gps_msg = self.mav.recv_match(type="GPS_RAW_INT", blocking=False)
            if not gps_msg:
                break
            self._last_gps_fix = gps_msg.fix_type
            if self._last_gps_fix < 5:
                log.warning(f"GPS fix degraded: fix_type={self._last_gps_fix}")
        return self._last_armed, self._last_mode

    def is_armed(self) -> bool:
        return self._last_armed

    def get_mode(self) -> str:
        return self._last_mode

    def get_altitude(self) -> float:
        return self._last_altitude

    def get_state_snapshot(self) -> dict:
        return {
            "armed": self._last_armed,
            "mode": self._last_mode,
            "altitude_m": self._last_altitude,
            "lat": self._last_lat / 1e7,
            "lon": self._last_lon / 1e7,
            "heading_deg": self._last_heading,
            "gps_fix": self._last_gps_fix,
            "timestamp": time.time(),
        }

    def set_geofence_radial(self, radius_m: float) -> None:
        self._geofence_radial = radius_m
        log.info(f"Radial geofence set: {radius_m}m from home")

    def set_geofence_square(self, half_side_m: float) -> None:
        if self._home_xy is None:
            log.error("Cannot set square geofence: home not captured")
            return
        home_lat_deg = self._home_xy[0] / 1e7
        home_lon_deg = self._home_xy[1] / 1e7
        dlat_deg = half_side_m / 111_300
        dlon_deg = half_side_m / (111_300 * math.cos(math.radians(home_lat_deg)))
        self._geofence_square = (
            int((home_lat_deg - dlat_deg) * 1e7),
            int((home_lat_deg + dlat_deg) * 1e7),
            int((home_lon_deg - dlon_deg) * 1e7),
            int((home_lon_deg + dlon_deg) * 1e7),
        )
        log.info(f"Square geofence set: ±{half_side_m}m from home")

    def get_geofence_violation(self) -> dict | None:
        if self._last_lat == 0 and self._last_lon == 0:
            return None
        if self._geofence_radial is not None and self._home_xy is not None:
            dlat = (self._last_lat - self._home_xy[0]) / 1e7
            dlon = (self._last_lon - self._home_xy[1]) / 1e7
            dist_m = math.sqrt(dlat**2 + dlon**2) * 111_300
            if dist_m > self._geofence_radial:
                return {
                    "type": "radial",
                    "dist_m": dist_m,
                    "limit_m": self._geofence_radial,
                    "excess_m": dist_m - self._geofence_radial,
                }
        if self._geofence_square is not None:
            min_lat, max_lat, min_lon, max_lon = self._geofence_square
            if not (
                min_lat <= self._last_lat <= max_lat
                and min_lon <= self._last_lon <= max_lon
            ):
                return {
                    "type": "square",
                    "lat": self._last_lat / 1e7,
                    "lon": self._last_lon / 1e7,
                    "bounds": {
                        "min_lat": min_lat / 1e7,
                        "max_lat": max_lat / 1e7,
                        "min_lon": min_lon / 1e7,
                        "max_lon": max_lon / 1e7,
                    },
                }
        return None

    def is_outside_geofence(self) -> bool:
        return self.get_geofence_violation() is not None

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

    def hover(
        self,
        altitude: float = config.MAVLINK_CRUISE_ALT,
        climb_rate: float = 0.5,
        timeout: float = config.MAVLINK_HOVER_TIMEOUT,
    ) -> bool:
        log.info(f"Hovering at {altitude:.1f}m (currently {self._last_altitude:.1f}m)")
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.update_state()
            if abs(self._last_altitude - altitude) < 0.1:
                log.info(f"Holding at {altitude:.1f}m")
                self.send_velocity(0.0, 0.0, 0.0)
                return True
            vz = climb_rate if self._last_altitude > altitude else -climb_rate
            self.send_velocity(0.0, 0.0, vz)
            time.sleep(0.05)
        log.error(f"Timed out reaching altitude {altitude:.1f}m")
        return False

    def go_to_coords(
        self,
        lat: float,
        lon: float,
        altitude: float = config.MAVLINK_CRUISE_ALT,
    ) -> None:
        log.info(f"Going to ({lat:.6f}, {lon:.6f}) @ {altitude:.1f}m")
        self.mav.mav.set_position_target_global_int_send(
            0,
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            0b0000_1111_1111_1000,
            int(lat * 1e7),
            int(lon * 1e7),
            altitude,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )

    def wait_armed(self, timeout: float = config.MAVLINK_ARM_TIMEOUT) -> bool:
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

    def wait_for_guided(self, timeout: float = config.MAVLINK_GUIDED_TIMEOUT) -> bool:
        log.info("Waiting for GUIDED mode")
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.update_state()
            if "GUIDED" in self.get_mode().upper():
                log.info("GUIDED mode confirmed")
                return True
            time.sleep(0.5)
        log.error("Timeout waiting for GUIDED")
        return False

    def takeoff(self, altitude: float = config.MAVLINK_CRUISE_ALT) -> bool:
        log.info(f"Takeoff to {altitude:.1f}m")
        self.mav.mav.command_long_send(
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            altitude,
        )
        ack = self.mav.recv_match(type="COMMAND_ACK", blocking=True, timeout=3)
        if not ack or ack.result != mavutil.mavlink.MAV_RESULT_ACCEPTED:
            log.error("Takeoff command not acknowledged")
            return False
        deadline = time.time() + config.MAVLINK_HOVER_TIMEOUT
        while time.time() < deadline:
            self.update_state()
            if self._last_altitude >= altitude * 0.95:
                log.info(f"Takeoff complete at {self._last_altitude:.1f}m")
                return True
            time.sleep(0.2)
        log.error("Takeoff timed out")
        return False

    def capture_home(
        self, timeout: float = config.MAVLINK_HOME_CAPTURE_TIMEOUT
    ) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.mav.messages.get("GLOBAL_POSITION_INT")
            if msg and (msg.lat != 0 or msg.lon != 0):
                self._home_xy = (msg.lat, msg.lon)
                log.info(f"Home captured: ({msg.lat / 1e7:.6f}, {msg.lon / 1e7:.6f})")
                return True
            time.sleep(0.2)
        log.warning("Home capture failed")
        return False

    def return_to_launch(self) -> bool:
        mode_id = self.mav.mode_mapping().get("RTL")
        if mode_id is None:
            log.error("RTL not in mode map, attempting fallback return")
            return self._fallback_return()
        self.mav.mav.set_mode_send(
            self.mav.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id,
        )
        ack = self.mav.recv_match(type="COMMAND_ACK", blocking=True, timeout=3)
        if ack and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
            log.info("RTL accepted by FC")
            return True
        log.error("RTL not acknowledged, attempting fallback return")
        return self._fallback_return()

    def _fallback_return(
        self,
        altitude: float = config.MAVLINK_CRUISE_ALT,
        timeout: float = config.MAVLINK_LAND_TIMEOUT,
    ) -> bool:
        if self._home_xy is None:
            log.error("No home captured, cannot fallback return")
            return False
        log.warning("Fallback return: navigating to home XY then landing")
        home_lat, home_lon = self._home_xy
        self.mav.mav.set_position_target_global_int_send(
            0,
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            0b0000_1111_1111_1000,
            int(home_lat),
            int(home_lon),
            altitude,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.update_state()
            msg = self.mav.messages.get("GLOBAL_POSITION_INT")
            if msg:
                dlat = (msg.lat - home_lat) / 1e7
                dlon = (msg.lon - home_lon) / 1e7
                dist_m = math.sqrt(dlat**2 + dlon**2) * 111_300
                if dist_m < 2.0:
                    log.info("Reached home XY, initiating land")
                    return self.land()
            time.sleep(0.5)
        log.error("Fallback return timed out")
        return False

    def land(self) -> bool:
        mode_id = self.mav.mode_mapping().get("LAND")
        if mode_id is None:
            log.error("LAND mode not found in mode map")
            return False
        self.mav.mav.set_mode_send(
            self.mav.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id,
        )
        ack = self.mav.recv_match(type="COMMAND_ACK", blocking=True, timeout=3)
        if ack and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
            log.info("LAND accepted by FC")
            return True
        log.error("LAND not acknowledged by FC")
        return False
