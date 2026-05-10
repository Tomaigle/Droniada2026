# TODO: CV verify — if ball visible at similar depth to _depth_before_grip = grip failed
import time
import logging
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional

from detector import Detection
import config

log = logging.getLogger(__name__)


class State(Enum):
    WAIT_ARM = auto()
    WAIT_START = auto()
    REVERSE_TO_BALLS = auto()
    SCAN_BALLS = auto()
    ALIGN_BALL = auto()
    LAND_ON_BALL = auto()
    GRIP = auto()
    LIFT_AND_VERIFY = auto()
    FLY_TO_BARRELS = auto()
    SCAN_BARRELS = auto()
    ALIGN_BARREL = auto()
    DROP = auto()
    RETURN_TO_BALLS = auto()
    COMPLETE = auto()
    ABORT = auto()


@dataclass
class BallRecord:
    colour: str
    cx: int
    cy: int
    depth: float
    picked_up: bool = False
    grip_retries: int = 0


class MissionController:
    def __init__(self, mav, gripper, camera, ball_det, barrel_det, operator_start):
        self.mav = mav
        self.gripper = gripper
        self.cam = camera
        self.ball_det = ball_det
        self.barrel_det = barrel_det
        self.operator_start = operator_start

        self.state = State.WAIT_ARM
        self._state_enter_time = time.time()
        self._mission_start_time = 0.0
        self._now = 0.0
        self._frame = None
        self._depth_frame = None

        self.ball_records: list[BallRecord] = []
        self.current_ball_idx: int = 0
        self._ball_zone_gps: Optional[tuple[float, float]] = None
        self._barrel_zone_gps: Optional[tuple[float, float]] = None
        self._barrels_scanned: bool = False
        self.delivered_count: int = 0

        self._travel_start_time = 0.0
        self._lift_start_alt = 0.0
        self._grip_time = 0.0
        self._depth_before_grip = 0.0
        self.holding_ball = False
        self.last_telemetry: dict = {}

    def run_step(self, frame, depth_frame) -> dict:
        self._now = time.time()
        self._frame = frame
        self._depth_frame = depth_frame

        if (
            self._mission_start_time
            and (self._now - self._mission_start_time) > config.MISSION_TIMEOUT_S
        ):
            log.error("Mission timeout — forced abort")
            self._transition(State.ABORT)

        handler = {
            State.WAIT_ARM: self._wait_arm,
            State.WAIT_START: self._wait_start,
            State.REVERSE_TO_BALLS: self._reverse_to_balls,
            State.SCAN_BALLS: self._scan_balls,
            State.ALIGN_BALL: self._align_ball,
            State.LAND_ON_BALL: self._land_on_ball,
            State.GRIP: self._grip,
            State.LIFT_AND_VERIFY: self._lift_and_verify,
            State.FLY_TO_BARRELS: self._fly_to_barrels,
            State.SCAN_BARRELS: self._scan_barrels,
            State.ALIGN_BARREL: self._align_barrel,
            State.DROP: self._drop,
            State.RETURN_TO_BALLS: self._return_to_balls,
            State.COMPLETE: self._complete,
            State.ABORT: self._abort,
        }[self.state]

        tel = handler()
        tel.update(
            {
                "state": self.state.name,
                "holding": self.holding_ball,
                "delivered": self.delivered_count,
                "queue": [b.colour for b in self.ball_records if not b.picked_up],
            }
        )
        self.last_telemetry = tel
        return tel

    def _wait_arm(self) -> dict:
        if self._just_entered():
            log.info("Waiting for RC arm…")
        if self.mav.is_armed():
            log.info("Armed — waiting for operator start")
            self._transition(State.WAIT_START)
        return {}

    def _wait_start(self) -> dict:
        if self._just_entered():
            log.info("Type 'start' + Enter to begin")
        if self.operator_start.is_set():
            self._mission_start_time = self._now
            self._transition(State.REVERSE_TO_BALLS)
        return {}

    def _reverse_to_balls(self) -> dict:
        if self._just_entered():
            log.info("Reversing to ball zone…")
            self._travel_start_time = self._now

        distance = (self._now - self._travel_start_time) * config.INITIAL_REVERSE_SPEED

        if distance < config.INITIAL_REVERSE_M:
            self.mav.send_velocity(-config.INITIAL_REVERSE_SPEED, 0.0, 0.0)
            return {"travel_m": distance}

        self.mav.hover()
        snap = self.mav.get_state_snapshot()
        self._ball_zone_gps = (snap["lat"], snap["lon"])
        log.info("Ball zone GPS: %.7f, %.7f", *self._ball_zone_gps)
        self._transition(State.SCAN_BALLS)
        return {"travel_m": distance}

    def _scan_balls(self) -> dict:
        if self._just_entered():
            self.mav.hover()

        all_dets = self.ball_det.detect(self._frame, self._depth_frame)
        for det in all_dets:
            if (
                not any(r.colour == det.colour for r in self.ball_records)
                and det.colour in config.PICKUP_ORDER
            ):
                log.info("Found %s ball @ %.2f m", det.colour, det.depth)
                self.ball_records.append(
                    BallRecord(colour=det.colour, cx=det.cx, cy=det.cy, depth=det.depth)
                )

        order_map = {c: i for i, c in enumerate(config.PICKUP_ORDER)}
        self.ball_records.sort(key=lambda r: order_map.get(r.colour, 99))

        next_idx = next(
            (i for i, r in enumerate(self.ball_records) if not r.picked_up), None
        )

        if next_idx is not None:
            self.current_ball_idx = next_idx
            log.info("Targeting %s", self.ball_records[next_idx].colour)
            self._transition(State.ALIGN_BALL)
        elif self._now - self._state_enter_time > config.SEARCH_TIMEOUT_S:
            log.warning("Scan timeout — no pickable balls, retrying")
            self._transition(State.SCAN_BALLS)

        return {
            "found": [r.colour for r in self.ball_records],
            "remaining": [r.colour for r in self.ball_records if not r.picked_up],
        }

    def _align_ball(self) -> dict:
        rec = self.ball_records[self.current_ball_idx]
        target = rec.colour

        if self._just_entered():
            log.info("Aligning over %s (retry %d)", target, rec.grip_retries)

        dets = self.ball_det.detect(
            self._frame, self._depth_frame, target_colour=target
        )

        if not dets:
            sweep_vy = _sweep_direction_from_history(
                self.ball_records, target, self.cam.width
            )
            self.mav.send_velocity(0.0, sweep_vy, 0.0)
            if self._now - self._state_enter_time > config.ALIGN_TIMEOUT_S:
                log.warning("Lost %s — re-scanning", target)
                self._transition(State.SCAN_BALLS)
            return {"sweep_vy": sweep_vy}

        det = dets[0]
        vx = float(
            np.clip(
                config.KP_XY * det.err_x_m, -config.MAX_SPEED_XY, config.MAX_SPEED_XY
            )
        )
        vy = float(
            np.clip(
                config.KP_XY * det.err_y_m, -config.MAX_SPEED_XY, config.MAX_SPEED_XY
            )
        )
        self.mav.send_velocity(vx, vy, 0.0)

        if _is_centered(det):
            self._depth_before_grip = det.depth
            self._transition(State.LAND_ON_BALL)

        return {"err_x": det.err_x_m, "err_y": det.err_y_m}

    def _land_on_ball(self) -> dict:
        if self._just_entered():
            log.info("Descending onto ball…")

        rec = self.ball_records[self.current_ball_idx]
        dets = self.ball_det.detect(
            self._frame, self._depth_frame, target_colour=rec.colour
        )

        if dets:
            det = dets[0]
            vx = float(
                np.clip(
                    config.KP_XY * det.err_x_m,
                    -config.MAX_SPEED_XY,
                    config.MAX_SPEED_XY,
                )
            )
            vy = float(
                np.clip(
                    config.KP_XY * det.err_y_m,
                    -config.MAX_SPEED_XY,
                    config.MAX_SPEED_XY,
                )
            )
            self.mav.send_velocity(vx, vy, config.LAND_SPEED_Z)
        else:
            self.mav.send_velocity(0.0, 0.0, config.LAND_SPEED_Z)

        snap = self.mav.get_state_snapshot()

        if snap["alt_m"] <= config.GRIP_LAND_ALT_M:
            self.mav.hover()
            log.info("At grip altitude %.2f m", snap["alt_m"])
            self._lift_start_alt = snap["alt_m"]
            self._transition(State.GRIP)
            return {"alt_m": snap["alt_m"]}

        if self._now - self._state_enter_time > config.LAND_TIMEOUT_S:
            log.warning("Land timeout — back to align")
            self._ascend_to_cruise()
            self._transition(State.ALIGN_BALL)

        return {"alt_m": snap["alt_m"]}

    def _grip(self) -> dict:
        if self._just_entered():
            self.gripper.close()
            self._grip_time = self._now
            log.info("Gripper closed — holding %.1f s", config.GRIP_HOLD_S)

        if self._now - self._grip_time >= config.GRIP_HOLD_S:
            self._transition(State.LIFT_AND_VERIFY)

        return {}

    def _lift_and_verify(self) -> dict:
        if self._just_entered():
            log.info("Lifting to verify…")

        self.mav.send_velocity(0.0, 0.0, -config.MAX_SPEED_Z)
        snap = self.mav.get_state_snapshot()
        lifted = snap["alt_m"] - self._lift_start_alt

        if lifted >= config.GRIP_VERIFY_ALT_M:
            self.mav.hover()
            log.info("Lifted %.2f m — grip assumed good", lifted)
            self.holding_ball = True
            rec = self.ball_records[self.current_ball_idx]
            rec.picked_up = True
            self._ascend_to_cruise()
            self._transition(State.FLY_TO_BARRELS)
            return {"lifted_m": lifted}

        if self._now - self._state_enter_time > config.GRIP_VERIFY_TIMEOUT_S:
            rec = self.ball_records[self.current_ball_idx]
            rec.grip_retries += 1
            log.warning(
                "Lift timeout — retry %d/%d", rec.grip_retries, config.MAX_GRIP_RETRIES
            )
            self.gripper.open()
            self.holding_ball = False
            self._ascend_to_cruise()

            if rec.grip_retries >= config.MAX_GRIP_RETRIES:
                log.error("Max retries for %s — skipping", rec.colour)
                rec.picked_up = True
                next_idx = next(
                    (i for i, r in enumerate(self.ball_records) if not r.picked_up),
                    None,
                )
                self._transition(
                    State.ALIGN_BALL if next_idx is not None else State.COMPLETE
                )
                if next_idx is not None:
                    self.current_ball_idx = next_idx
            else:
                self._transition(State.ALIGN_BALL)

        return {"lifted_m": lifted}

    def _fly_to_barrels(self) -> dict:
        if self._just_entered():
            log.info("Flying to barrel zone…")
            self._travel_start_time = self._now

        distance = (self._now - self._travel_start_time) * config.BARREL_SEARCH_SPEED

        if distance < config.BARREL_SEARCH_FWD_M:
            self.mav.send_velocity(config.BARREL_SEARCH_SPEED, 0.0, 0.0)
            _gps_sanity_check(self.mav, self._barrel_zone_gps, "barrel zone")
            return {"travel_m": distance}

        self.mav.hover()
        if not self._barrels_scanned:
            snap = self.mav.get_state_snapshot()
            self._barrel_zone_gps = (snap["lat"], snap["lon"])
            log.info("Barrel zone GPS: %.7f, %.7f", *self._barrel_zone_gps)
            self._transition(State.SCAN_BARRELS)
        else:
            self._transition(State.ALIGN_BARREL)
        return {"travel_m": distance}

    def _scan_barrels(self) -> dict:
        if self._just_entered():
            self.mav.hover()

        dets = sorted(
            self.barrel_det.detect(self._frame, self._depth_frame), key=lambda d: d.cx
        )

        if len(dets) >= 3:
            log.info(
                "3 barrels: L cx=%d C cx=%d R cx=%d", dets[0].cx, dets[1].cx, dets[2].cx
            )
            self._barrels_scanned = True
            self._transition(State.ALIGN_BARREL)
        elif self._now - self._state_enter_time > config.SEARCH_TIMEOUT_S:
            if dets:
                log.warning("Scan timeout — %d barrel(s), proceeding", len(dets))
                self._barrels_scanned = True
                self._transition(State.ALIGN_BARREL)
            else:
                log.warning("No barrels — sweeping and retrying")
                self.mav.send_velocity(0.0, config.BARREL_SWEEP_SPEED, 0.0)
                self._transition(State.SCAN_BARRELS)

        return {"barrels_seen": len(dets)}

    def _align_barrel(self) -> dict:
        target_idx = self.delivered_count

        if self._just_entered():
            log.info("Targeting barrel idx %d (L/C/R)", target_idx)

        dets = sorted(
            self.barrel_det.detect(self._frame, self._depth_frame), key=lambda d: d.cx
        )

        if len(dets) <= target_idx:
            sweep_vy = _barrel_sweep_direction(dets, target_idx, self.cam.width)
            self.mav.send_velocity(0.0, sweep_vy, 0.0)
            if self._now - self._state_enter_time > config.ALIGN_TIMEOUT_S:
                log.warning("Barrel align timeout — re-scanning")
                self._barrels_scanned = False
                self._transition(State.SCAN_BARRELS)
            return {"barrels_seen": len(dets), "sweep_vy": sweep_vy}

        det = dets[target_idx]
        vx, vy, vz = _compute_velocity(det, target_depth=config.DROP_HOVER_M)
        self.mav.send_velocity(vx, vy, vz)

        if _is_centered(det) and abs(det.depth - config.DROP_HOVER_M) < 0.15:
            self._transition(State.DROP)

        return {"barrel_idx": target_idx, "err_x": det.err_x_m, "err_y": det.err_y_m}

    def _drop(self) -> dict:
        if self._just_entered():
            self.mav.hover()
            self.gripper.open()
            self.holding_ball = False
            self.delivered_count += 1
            log.info("Dropped — delivered %d", self.delivered_count)

        if self._now - self._state_enter_time > 1.5:
            remaining = [r for r in self.ball_records if not r.picked_up]
            if remaining:
                self._ascend_to_cruise()
                self._transition(State.RETURN_TO_BALLS)
            else:
                self._transition(State.COMPLETE)

        return {"delivered": self.delivered_count}

    def _return_to_balls(self) -> dict:
        if self._just_entered():
            log.info("Returning to ball zone…")
            self._travel_start_time = self._now

        distance = (self._now - self._travel_start_time) * config.BARREL_SEARCH_SPEED

        if distance < config.BARREL_SEARCH_FWD_M:
            self.mav.send_velocity(-config.BARREL_SEARCH_SPEED, 0.0, 0.0)
            _gps_sanity_check(self.mav, self._ball_zone_gps, "ball zone")
            return {"travel_m": distance}

        self.mav.hover()
        log.info("Back at ball zone")
        self._transition(State.SCAN_BALLS)
        return {"travel_m": distance}

    def _complete(self) -> dict:
        if self._just_entered():
            log.info("Mission complete — RTL")
            self.mav.set_mode("RTL")
        return {}

    def _abort(self) -> dict:
        if self._just_entered():
            log.error("ABORT — gripper open, RTL")
            self.gripper.open()
            self.mav.hover()
            time.sleep(0.5)
            self.mav.set_mode("RTL")
        return {}

    def _transition(self, new: State) -> None:
        log.info("%-20s → %s", self.state.name, new.name)
        self.state = new
        self._state_enter_time = time.time()

    def _just_entered(self) -> bool:
        return (time.time() - self._state_enter_time) < 0.05

    def _ascend_to_cruise(self) -> None:
        log.info("Ascending…")
        self.mav.send_velocity(0.0, 0.0, -config.MAX_SPEED_Z)
        time.sleep(config.ASCENT_DURATION_S)
        self.mav.hover()


def _sweep_direction_from_history(ball_records, target_colour, frame_width) -> float:
    for rec in ball_records:
        if rec.colour == target_colour:
            return (
                config.BARREL_SWEEP_SPEED
                if rec.cx >= frame_width // 2
                else -config.BARREL_SWEEP_SPEED
            )
    return config.BARREL_SWEEP_SPEED


def _barrel_sweep_direction(visible_dets, frame_width) -> float:
    if not visible_dets:
        return config.BARREL_SWEEP_SPEED
    avg_cx = sum(d.cx for d in visible_dets) / len(visible_dets)
    return (
        config.BARREL_SWEEP_SPEED
        if avg_cx < frame_width // 2
        else -config.BARREL_SWEEP_SPEED
    )


def _gps_sanity_check(mav, saved_gps, label) -> None:
    if saved_gps is None:
        return
    snap = mav.get_state_snapshot()
    lat, lon = snap["lat"], snap["lon"]
    if lat == 0.0 and lon == 0.0:
        return
    dlat = abs(lat - saved_gps[0])
    dlon = abs(lon - saved_gps[1])
    if dlat > config.GPS_SANITY_DEG_THRESH or dlon > config.GPS_SANITY_DEG_THRESH:
        log.warning("GPS drift vs %s: Δlat=%.6f Δlon=%.6f", label, dlat, dlon)


def _compute_velocity(
    det: Detection, target_depth: float
) -> tuple[float, float, float]:
    vx = float(
        np.clip(config.KP_XY * det.err_x_m, -config.MAX_SPEED_XY, config.MAX_SPEED_XY)
    )
    vy = float(
        np.clip(config.KP_XY * det.err_y_m, -config.MAX_SPEED_XY, config.MAX_SPEED_XY)
    )
    depth_err = det.depth - target_depth
    vz = (
        float(np.clip(config.KP_Z * depth_err, 0.0, config.MAX_SPEED_Z))
        if (_is_centered(det) and depth_err > 0.05)
        else 0.0
    )
    return vx, vy, vz


def _is_centered(det: Detection) -> bool:
    return (
        abs(det.err_x_m) < config.CENTER_THRESH_M
        and abs(det.err_y_m) < config.CENTER_THRESH_M
    )
