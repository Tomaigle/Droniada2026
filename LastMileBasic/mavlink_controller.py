import time
import queue
import threading
import logging
from typing import Optional
from dataclasses import dataclass

from pymavlink import mavutil
import config

log = logging.getLogger(__name__)


@dataclass
class _CmdVelocity:
    vx: float
    vy: float
    vz: float


@dataclass
class _CmdMode:
    mode: str


@dataclass
class _CmdTakeoff:
    alt_m: float


@dataclass
class _CmdStop:
    pass


class MAVLinkController:
    def __init__(
        self, port: str = config.MAVLINK_PORT, baud: int = config.MAVLINK_BAUD
    ):
        self.port = port
        self.baud = baud
        self._mav: Optional[mavutil.mavfile] = None
        self._cmd_q: queue.Queue = queue.Queue(maxsize=20)
        self._running = False
        self._io_thread: Optional[threading.Thread] = None
        self._state_lock = threading.Lock()
        self._state: dict = {
            "armed": False,
            "mode": "UNKNOWN",
            "alt_m": 0.0,
            "lat": 0.0,
            "lon": 0.0,
            "vx_ms": 0.0,
            "vy_ms": 0.0,
            "vz_ms": 0.0,
            "geofence_breach": False,
            "fc_connected": False,
        }

    def connect(self, timeout: float = 30.0) -> None:
        log.info("Connecting to FC on %s @ %d baud …", self.port, self.baud)
        self._mav = mavutil.mavlink_connection(
            self.port, baud=self.baud, source_system=255, source_component=0
        )
        if not self._mav.wait_heartbeat(timeout=timeout):
            raise ConnectionError(
                "No heartbeat received — check wiring and SERIAL_PROTOCOL=2 on FC."
            )
        log.info(
            "FC connected — sysid=%d compid=%d",
            self._mav.target_system,
            self._mav.target_component,
        )
        with self._state_lock:
            self._state["fc_connected"] = True

        for stream_id, rate_hz in (
            (mavutil.mavlink.MAV_DATA_STREAM_POSITION, 10),
            (mavutil.mavlink.MAV_DATA_STREAM_EXTRA1, 10),
        ):
            self._mav.mav.request_data_stream_send(
                self._mav.target_system,
                self._mav.target_component,
                stream_id,
                rate_hz,
                1,
            )
        log.info("Telemetry streams requested at 10 Hz")
        self._running = True
        self._io_thread = threading.Thread(
            target=self._io_loop, name="mav-io", daemon=True
        )
        self._io_thread.start()

    def disconnect(self) -> None:
        self._cmd_q.put(_CmdStop())
        self._running = False
        if self._io_thread:
            self._io_thread.join(timeout=3.0)
        if self._mav:
            self._mav.close()
        log.info("MAVLink disconnected")

    def send_velocity(self, vx: float, vy: float, vz: float) -> None:
        self._enqueue(_CmdVelocity(vx, vy, vz))

    def hover(self) -> None:
        self._enqueue(_CmdVelocity(0.0, 0.0, 0.0))

    def land(self) -> None:
        self._enqueue(_CmdMode("LAND"))

    def takeoff(self, alt_m: float = config.CRUISE_ALT_M) -> None:
        self._enqueue(_CmdTakeoff(alt_m))

    def set_mode(self, mode: str) -> None:
        self._enqueue(_CmdMode(mode))

    def _enqueue(self, cmd) -> None:
        try:
            self._cmd_q.put_nowait(cmd)
        except queue.Full:
            log.warning("Command queue full — dropping %s", type(cmd).__name__)

    def get_state_snapshot(self) -> dict:
        with self._state_lock:
            return dict(self._state)

    def is_geofence_breached(self) -> bool:
        with self._state_lock:
            return self._state["geofence_breach"]

    def is_armed(self) -> bool:
        with self._state_lock:
            return self._state["armed"]

    def _io_loop(self) -> None:
        last_hb_t = 0.0
        last_fence_t = 0.0
        HB_INTERVAL = 1.0
        FENCE_HZ = 5.0

        log.info("MAVLink I/O thread started")

        while self._running:
            now = time.monotonic()

            try:
                while True:
                    cmd = self._cmd_q.get_nowait()
                    if isinstance(cmd, _CmdStop):
                        log.info("MAVLink I/O thread stopping")
                        return
                    self._execute(cmd)
            except queue.Empty:
                pass

            if now - last_hb_t >= HB_INTERVAL:
                self._mav.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_GCS,
                    mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                    0,
                    0,
                    0,
                )
                last_hb_t = now

            msg = self._mav.recv_match(blocking=False)
            if msg:
                self._handle_msg(msg)

            if now - last_fence_t >= 1.0 / FENCE_HZ:
                self._check_geofence()
                last_fence_t = now

            time.sleep(0.005)

        log.info("MAVLink I/O thread exited")

    def _execute(self, cmd) -> None:
        if isinstance(cmd, _CmdVelocity):
            self._mav.mav.set_position_target_local_ned_send(
                0,
                self._mav.target_system,
                self._mav.target_component,
                mavutil.mavlink.MAV_FRAME_BODY_NED,
                0b0000_1111_1100_0111,
                0,
                0,
                0,
                cmd.vx,
                cmd.vy,
                cmd.vz,
                0,
                0,
                0,
                0,
                0,
            )

        elif isinstance(cmd, _CmdMode):
            mode_id = self._mav.mode_mapping().get(cmd.mode.upper())
            if mode_id is None:
                log.error("Mode %r not in FC mode map", cmd.mode)
                return
            self._mav.mav.set_mode_send(
                self._mav.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode_id,
            )
            log.info("Mode → %s", cmd.mode)

        elif isinstance(cmd, _CmdTakeoff):
            self._mav.mav.command_long_send(
                self._mav.target_system,
                self._mav.target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                cmd.alt_m,
            )
            log.info("Takeoff → %.1f m", cmd.alt_m)

    def _handle_msg(self, msg) -> None:
        mtype = msg.get_type()

        if mtype == "HEARTBEAT":
            armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            mode = mavutil.mode_string_v10(msg)
            with self._state_lock:
                self._state["armed"] = armed
                self._state["mode"] = mode

        elif mtype == "GLOBAL_POSITION_INT":
            with self._state_lock:
                self._state["alt_m"] = msg.relative_alt / 1000.0
                self._state["lat"] = msg.lat / 1e7
                self._state["lon"] = msg.lon / 1e7
                self._state["vx_ms"] = msg.vx / 100.0
                self._state["vy_ms"] = msg.vy / 100.0
                self._state["vz_ms"] = msg.vz / 100.0

    def _check_geofence(self) -> None:
        with self._state_lock:
            already = self._state["geofence_breach"]
            lat = self._state["lat"]
            lon = self._state["lon"]

        if already or (lat == 0.0 and lon == 0.0):
            return

        inside = (
            config.GEOFENCE_LAT_MIN <= lat <= config.GEOFENCE_LAT_MAX
            and config.GEOFENCE_LON_MIN <= lon <= config.GEOFENCE_LON_MAX
        )

        if not inside:
            log.critical("GEOFENCE BREACH — lat=%.7f lon=%.7f", lat, lon)
            self._mav.mav.set_position_target_local_ned_send(
                0,
                self._mav.target_system,
                self._mav.target_component,
                mavutil.mavlink.MAV_FRAME_BODY_NED,
                0b0000_1111_1100_0111,
                0,
                0,
                0,
                0.0,
                0.0,
                0.0,
                0,
                0,
                0,
                0,
                0,
            )
            with self._state_lock:
                self._state["geofence_breach"] = True


class MockMAVLinkController:
    def __init__(self):
        self._alt = config.CRUISE_ALT_M
        self._breach = False

    def connect(self, **_) -> None:
        log.info("[MOCK MAV] Connected")

    def disconnect(self) -> None:
        log.info("[MOCK MAV] Disconnected")

    def send_velocity(self, vx: float, vy: float, vz: float) -> None:
        log.debug("[MOCK MAV] VEL vx=%+.2f vy=%+.2f vz=%+.2f", vx, vy, vz)

    def hover(self) -> None:
        log.debug("[MOCK MAV] HOVER")

    def land(self) -> None:
        log.info("[MOCK MAV] LAND")

    def takeoff(self, alt_m: float = config.CRUISE_ALT_M) -> None:
        log.info("[MOCK MAV] TAKEOFF → %.1f m", alt_m)
        self._alt = alt_m

    def set_mode(self, mode: str) -> None:
        log.info("[MOCK MAV] MODE → %s", mode)

    def get_state_snapshot(self) -> dict:
        return {
            "armed": True,
            "mode": "GUIDED",
            "alt_m": self._alt,
            "lat": config.GEOFENCE_LAT_MIN + 0.00005,
            "lon": config.GEOFENCE_LON_MIN + 0.00005,
            "vx_ms": 0.0,
            "vy_ms": 0.0,
            "vz_ms": 0.0,
            "geofence_breach": self._breach,
            "fc_connected": True,
        }

    def is_geofence_breached(self) -> bool:
        return self._breach

    def is_armed(self) -> bool:
        return True
