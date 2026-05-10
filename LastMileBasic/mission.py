import time
import logging
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional

from mavlink_controller import MAVLinkController
from gripper import Gripper
from detector import Detection, BallDetector, BarrelDetector, RealsenseCamera
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
    VERIFY_GRIP = auto()
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
    def __init__(
        self,
        mav: MAVLinkController,
        gripper: Gripper,
        camera: RealsenseCamera,
        ball_det: BallDetector,
        barrel_det: BarrelDetector,
        operator_start,
    ):
        self.mav = mav
        self.gripper = gripper
        self.cam = camera
        self.ball_det = ball_det
        self.barrel_det = barrel_det
        self.operator_start = operator_start

        self.state = State.WAIT_ARM
        self._state_enter_time = time.time()
        self._mission_start_time = 0.0

        self.ball_records: list[BallRecord] = []
        self.current_ball_idx: int = 0
        self._ball_zone_gps: Optional[tuple[float, float]] = None

        self._barrel_zone_gps: Optional[tuple[float, float]] = None
        self._barrels_scanned: bool = False
        self.delivered_count: int = 0

        self._travel_start_time = 0.0

        self._grip_time = 0.0
        self._depth_before_grip = 0.0
        self.holding_ball = False

        self.last_telemetry: dict = {}

    def run_step(self, frame, depth_frame) -> dict:
        now = time.time()

        if (
            self._mission_start_time
            and (now - self._mission_start_time) > config.MISSION_TIMEOUT_S
        ):
            log.error("Mission timeout — aborting")
            self._transition(State.ABORT)

        handler = {
            State.WAIT_ARM: self._wait_arm,
            State.WAIT_START: self._wait_start,
            State.REVERSE_TO_BALLS: self._reverse_to_balls,
            State.SCAN_BALLS: self._scan_balls,
            State.ALIGN_BALL: self._align_ball,
            State.LAND_ON_BALL: self._land_on_ball,
            State.GRIP: self._grip,
            State.VERIFY_GRIP: self._verify_grip,
            State.FLY_TO_BARRELS: self._fly_to_barrels,
            State.SCAN_BARRELS: self._scan_barrels,
            State.ALIGN_BARREL: self._align_barrel,
            State.DROP: self._drop,
            State.RETURN_TO_BALLS: self._return_to_balls,
            State.COMPLETE: self._complete,
            State.ABORT: self._abort,
        }[self.state]

        tel = handler(frame, depth_frame, now)
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
            self._mission_start_time = now
            log.info("Mission started")
            self._transition(State.REVERSE_TO_BALLS)
        return {}

    def _reverse_to_balls(self, now) -> dict:
        if self._just_entered():
            log.info("Reversing %.1f m to ball zone", config.INITIAL_REVERSE_M)
            self._travel_start_time = now

        elapsed = now - self._travel_start_time
        distance = elapsed * config.INITIAL_REVERSE_SPEED

        if distance < config.INITIAL_REVERSE_M:
            self.mav.send_velocity(-config.INITIAL_REVERSE_SPEED, 0.0, 0.0)
        else:
            self.mav.hover()
            snap = self.mav.get_state_snapshot()
            self._ball_zone_gps = (snap["lat"], snap["lon"])
            log.info("Ball zone GPS saved: %.7f, %.7f", *self._ball_zone_gps)
            self._transition(State.SCAN_BALLS)

        return {"travel_m": distance}

    def _scan_balls(self, frame, depth_frame, now) -> dict:
        if self._just_entered():
            log.info("Scanning for remaining balls…")
            self.mav.hover()

        all_dets = self.ball_det.detect(frame, depth_frame)

        for det in all_dets:
            already = any(r.colour == det.colour for r in self.ball_records)
            if not already and det.colour in config.PICKUP_ORDER:
                log.info("  Found %s ball @ %.2f m", det.colour, det.depth)
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
            log.info("Targeting %s ball", self.ball_records[next_idx].colour)
            self._transition(State.ALIGN_BALL)
        elif now - self._state_enter_time > config.SEARCH_TIMEOUT_S:
            if next_idx is not None:
                self.current_ball_idx = next_idx
                self._transition(State.ALIGN_BALL)
            else:
                log.error("No pickable balls found — abort")
                self._transition(State.ABORT)

        return {
            "found": [r.colour for r in self.ball_records],
            "remaining": [r.colour for r in self.ball_records if not r.picked_up],
        }

    def _align_ball(self, frame, depth_frame, now) -> dict:
        rec = self.ball_records[self.current_ball_idx]
        target = rec.colour

        if self._just_entered():
            log.info("Aligning over %s ball (retry %d)", target, rec.grip_retries)

        dets = self.ball_det.detect(frame, depth_frame, target_colour=target)

        if not dets:
            self.mav.send_velocity(0.0, 0.05, 0.0)
            if now - self._state_enter_time > config.ALIGN_TIMEOUT_S:
                log.warning("Lost %s — re-scanning", target)
                self._transition(State.SCAN_BALLS)
            return {}

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
            log.info("Centred over %s — descending", target)
            self._transition(State.LAND_ON_BALL)

        return {"err_x": det.err_x_m, "err_y": det.err_y_m}

    def _land_on_ball(self, frame, depth_frame, now) -> dict:
        if self._just_entered():
            log.info("Descending onto ball…")

        rec = self.ball_records[self.current_ball_idx]
        dets = self.ball_det.detect(frame, depth_frame, target_colour=rec.colour)

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
            log.info("On ground (%.2f m) — gripping", snap["alt_m"])
            self._transition(State.GRIP)

        if now - self._state_enter_time > config.LAND_TIMEOUT_S:
            log.warning("Land timeout — aborting")
            self._transition(State.ABORT)

        return {"alt_m": snap.get("alt_m", 0.0)}

    def _grip(self, now) -> dict:
        if self._just_entered():
            self.gripper.close()
            self._grip_time = now
            log.info("Gripper closed — holding %.1f s", config.GRIP_HOLD_S)

        if now - self._grip_time >= config.GRIP_HOLD_S:
            self._transition(State.VERIFY_GRIP)

        return {}

    def _verify_grip(self, frame, depth_frame, now) -> dict:
        elapsed = now - self._state_enter_time
        if elapsed < config.GRIP_VERIFY_WAIT_S:
            return {}

        rec = self.ball_records[self.current_ball_idx]
        dets = self.ball_det.detect(frame, depth_frame, target_colour=rec.colour)

        grip_ok = (
            not dets
            or abs(dets[0].depth - self._depth_before_grip)
            > config.GRIP_VERIFY_DEPTH_DELTA
        )

        if grip_ok:
            log.info("Grip CONFIRMED — ascending then delivering %s", rec.colour)
            self.holding_ball = True
            rec.picked_up = True
            self._ascend_to_cruise()
            self._transition(State.FLY_TO_BARRELS)
        else:
            rec.grip_retries += 1
            log.warning(
                "Grip FAILED — retry %d/%d", rec.grip_retries, config.MAX_GRIP_RETRIES
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
                if next_idx is not None:
                    self.current_ball_idx = next_idx
                    self._transition(State.ALIGN_BALL)
                else:
                    self._transition(State.COMPLETE)
            else:
                self._transition(State.ALIGN_BALL)

        return {}

    def _fly_to_barrels(self, now) -> dict:
        if self._just_entered():
            log.info("Flying to barrel zone %.1f m", config.BARREL_SEARCH_FWD_M)
            self._travel_start_time = now

        elapsed = now - self._travel_start_time
        distance = elapsed * config.BARREL_SEARCH_SPEED

        if distance < config.BARREL_SEARCH_FWD_M:
            self.mav.send_velocity(config.BARREL_SEARCH_SPEED, 0.0, 0.0)
        else:
            self.mav.hover()
            if not self._barrels_scanned:
                snap = self.mav.get_state_snapshot()
                self._barrel_zone_gps = (snap["lat"], snap["lon"])
                log.info("Barrel zone GPS saved: %.7f, %.7f", *self._barrel_zone_gps)
                self._transition(State.SCAN_BARRELS)
            else:
                log.info("Barrel zone known — aligning directly")
                self._transition(State.ALIGN_BARREL)

        return {"travel_m": distance}

    def _scan_barrels(self, frame, depth_frame, now) -> dict:
        if self._just_entered():
            log.info("Scanning for barrels…")
            self.mav.hover()

        dets = sorted(self.barrel_det.detect(frame, depth_frame), key=lambda d: d.cx)

        if len(dets) >= 3:
            log.info(
                "All 3 barrels visible — L cx=%d, C cx=%d, R cx=%d",
                dets[0].cx,
                dets[1].cx,
                dets[2].cx,
            )
            self._barrels_scanned = True
            self._transition(State.ALIGN_BARREL)
        elif now - self._state_enter_time > config.SEARCH_TIMEOUT_S:
            if dets:
                log.warning(
                    "Scan timeout — %d barrel(s) visible. Proceeding.", len(dets)
                )
                self._barrels_scanned = True
                self._transition(State.ALIGN_BARREL)
            else:
                log.error("No barrels — abort")
                self._transition(State.ABORT)

        return {"barrels_seen": len(dets)}

    def _align_barrel(self, frame, depth_frame, now) -> dict:
        target_idx = self.delivered_count

        if self._just_entered():
            log.info("Targeting barrel %d (idx %d by cx)", target_idx + 1, target_idx)

        dets = sorted(self.barrel_det.detect(frame, depth_frame), key=lambda d: d.cx)

        if len(dets) <= target_idx:
            self.mav.send_velocity(0.0, config.BARREL_SWEEP_SPEED, 0.0)
            log.debug(
                "Only %d barrels visible, need %d — sweeping", len(dets), target_idx + 1
            )
            if now - self._state_enter_time > config.ALIGN_TIMEOUT_S:
                log.warning("Barrel align timeout — re-scanning")
                self._barrels_scanned = False
                self._transition(State.SCAN_BARRELS)
            return {"barrels_seen": len(dets), "sweeping": True}

        det = dets[target_idx]
        vx, vy, vz = _compute_velocity(det, target_depth=config.DROP_HOVER_M)
        self.mav.send_velocity(vx, vy, vz)

        if _is_centered(det) and abs(det.depth - config.DROP_HOVER_M) < 0.15:
            log.info("Over barrel %d — dropping", target_idx + 1)
            self._transition(State.DROP)

        return {"barrel_idx": target_idx, "err_x": det.err_x_m, "err_y": det.err_y_m}

    def _drop(self, frame, depth_frame, now) -> dict:
        if self._just_entered():
            self.mav.hover()
            self.gripper.open()
            self.holding_ball = False
            self.delivered_count += 1
            log.info("Dropped — total delivered: %d", self.delivered_count)

        if now - self._state_enter_time > 1.5:
            remaining = [r for r in self.ball_records if not r.picked_up]
            if remaining:
                self._ascend_to_cruise()
                self._transition(State.RETURN_TO_BALLS)
            else:
                log.info("All balls delivered")
                self._transition(State.COMPLETE)

        return {"delivered": self.delivered_count}

    def _return_to_balls(self, now) -> dict:
        if self._just_entered():
            log.info("Returning to ball zone…")
            self._travel_start_time = now

        elapsed = now - self._travel_start_time
        distance = elapsed * config.BARREL_SEARCH_SPEED

        if distance < config.BARREL_SEARCH_FWD_M:
            self.mav.send_velocity(-config.BARREL_SEARCH_SPEED, 0.0, 0.0)
        else:
            self.mav.hover()
            log.info("Back at ball zone")
            self._transition(State.SCAN_BALLS)

        return {"travel_m": distance}

    def _complete(self) -> dict:
        if self._just_entered():
            log.info("COMPLETE — landing")
            self.mav.land()
        return {}

    def _abort(self) -> dict:
        if self._just_entered():
            log.error("ABORT — opening gripper, landing")
            self.gripper.open()
            self.mav.hover()
            time.sleep(0.5)
            self.mav.land()
        return {}

    def _transition(self, new: State) -> None:
        log.info("%-20s → %s", self.state.name, new.name)
        self.state = new
        self._state_enter_time = time.time()

    def _just_entered(self) -> bool:
        return (time.time() - self._state_enter_time) < 0.05

    def _ascend_to_cruise(self) -> None:
        log.info("Ascending (%.1f s)", config.ASCENT_DURATION_S)
        self.mav.send_velocity(0.0, 0.0, -config.MAX_SPEED_Z)
        time.sleep(config.ASCENT_DURATION_S)
        self.mav.hover()


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
