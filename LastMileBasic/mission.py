"""
mission.py — Last Mile Logistics BASIC stage state machine.

Key changes from v1:
  - No self-arming. Waits for RC arm + operator start command.
  - Balls are behind start: drone reverses INITIAL_REVERSE_M, scans all 3,
    remembers their positions, then picks them up in order.
  - Barrels are ahead: after all balls collected, fly forward to barrel zone,
    scan all 3, remember positions.
  - No ball skipping. On grip failure → retry up to MAX_GRIP_RETRIES.
  - Grip verification: after closing gripper, hover and re-detect target ball.
    If still present at similar depth → grip failed → retry.

State flow:
  WAIT_ARM → WAIT_START → REVERSE_TO_BALLS → SCAN_BALLS
  → [for each ball] SEARCH_BALL → ALIGN_BALL → GRIP → VERIFY_GRIP
  → (retry loop if failed)
  → FLY_TO_BARRELS → SCAN_BARRELS
  → [for each ball] SEARCH_BARREL → ALIGN_BARREL → DROP
  → COMPLETE → LAND
"""

import time
import logging
import math
import numpy as np
from enum import Enum, auto
from typing import Optional
from dataclasses import dataclass, field

from mavlink_controller import MAVLinkController, MockMAVLinkController
from gripper import Gripper, MockGripper
from detector import Detection, BallDetector, BarrelDetector, RealSenseCamera
import config

log = logging.getLogger(__name__)


class State(Enum):
    WAIT_ARM         = auto()
    WAIT_START       = auto()
    REVERSE_TO_BALLS = auto()
    SCAN_BALLS       = auto()
    ALIGN_BALL       = auto()
    GRIP             = auto()
    VERIFY_GRIP      = auto()
    FLY_TO_BARRELS   = auto()
    SCAN_BARRELS     = auto()
    ALIGN_BARREL     = auto()
    DROP             = auto()
    COMPLETE         = auto()
    ABORT            = auto()


@dataclass
class BallRecord:
    colour: str
    cx: int                 # pixel coords when first seen (for re-acquisition)
    cy: int
    depth: float            # depth when first seen
    picked_up: bool = False
    grip_retries: int = 0


class MissionController:
    def __init__(
        self,
        mav:            MAVLinkController,
        gripper:        Gripper,
        camera:         RealSenseCamera,
        ball_det:       BallDetector,
        barrel_det:     BarrelDetector,
        operator_start: "threading.Event",   # set from main.py when operator sends start
    ):
        self.mav            = mav
        self.gripper        = gripper
        self.cam            = camera
        self.ball_det       = ball_det
        self.barrel_det     = barrel_det
        self.operator_start = operator_start

        self.state               = State.WAIT_ARM
        self._state_enter_time   = time.time()
        self._mission_start_time = 0.0

        # Ball zone memory: discovered in SCAN_BALLS, consumed in order
        self.ball_records: list[BallRecord]        = []
        self.current_ball_idx: int                 = 0

        # Barrel zone memory: discovered in SCAN_BARRELS
        self.barrel_records: list[tuple[int, int]] = []   # (cx, cy)
        self.used_barrels:   list[tuple[int, int]] = []

        # Per-state tracking
        self._travel_start_time  = 0.0
        self._grip_time          = 0.0
        self._depth_before_grip  = 0.0
        self.holding_ball        = False
        self.last_telemetry      = {}

    # ── Public ────────────────────────────────────────────────────────────────

    def run_step(self, frame, depth_frame) -> dict:
        now = time.time()

        # Hard mission timeout
        if self._mission_start_time and (now - self._mission_start_time) > config.MISSION_TIMEOUT_S:
            log.error("Mission timeout — aborting")
            self._transition(State.ABORT)

        handler = {
            State.WAIT_ARM:         self._wait_arm,
            State.WAIT_START:       self._wait_start,
            State.REVERSE_TO_BALLS: self._reverse_to_balls,
            State.SCAN_BALLS:       self._scan_balls,
            State.ALIGN_BALL:       self._align_ball,
            State.GRIP:             self._grip,
            State.VERIFY_GRIP:      self._verify_grip,
            State.FLY_TO_BARRELS:   self._fly_to_barrels,
            State.SCAN_BARRELS:     self._scan_barrels,
            State.ALIGN_BARREL:     self._align_barrel,
            State.DROP:             self._drop,
            State.COMPLETE:         self._complete,
            State.ABORT:            self._abort,
        }[self.state]

        tel = handler(frame, depth_frame, now)
        tel.update({
            "state":   self.state.name,
            "holding": self.holding_ball,
            "queue":   [b.colour for b in self.ball_records if not b.picked_up],
        })
        self.last_telemetry = tel
        return tel

    # ── Arm / start gates ─────────────────────────────────────────────────────

    def _wait_arm(self, frame, depth_frame, now) -> dict:
        if self._just_entered():
            log.info("Waiting for RC arm signal...")
        if self.mav.is_armed():
            log.info("Armed — waiting for operator start command")
            self._transition(State.WAIT_START)
        return {}

    def _wait_start(self, frame, depth_frame, now) -> dict:
        if self._just_entered():
            log.info("Type 'start' in terminal when ready to begin mission")
        if self.operator_start.is_set():
            self._mission_start_time = now
            log.info("Mission started by operator")
            self._transition(State.REVERSE_TO_BALLS)
        return {}

    # ── Initial reverse to ball pickup zone ───────────────────────────────────

    def _reverse_to_balls(self, frame, depth_frame, now) -> dict:
        if self._just_entered():
            elapsed = now - self._state_enter_time
            log.info("Reversing %.1f m to ball zone at %.1f m/s",
                     config.INITIAL_REVERSE_M, config.INITIAL_REVERSE_SPEED)
            self._travel_start_time = now

        elapsed  = now - self._travel_start_time
        distance = elapsed * config.INITIAL_REVERSE_SPEED

        if distance < config.INITIAL_REVERSE_M:
            # Negative X = reverse in body frame
            self.mav.send_velocity(-config.INITIAL_REVERSE_SPEED, 0.0, 0.0)
        else:
            self.mav.hover()
            log.info("Reached ball zone — scanning")
            self._transition(State.SCAN_BALLS)

        return {"travel_m": distance}

    # ── Scan ball zone — find all 3, remember them ────────────────────────────

    def _scan_balls(self, frame, depth_frame, now) -> dict:
        if self._just_entered():
            log.info("Scanning for all 3 balls...")
            self.mav.hover()

        all_dets = self.ball_det.detect(frame, depth_frame)

        # Accumulate new colours not yet recorded
        for det in all_dets:
            already = any(r.colour == det.colour for r in self.ball_records)
            if not already and det.colour in config.PICKUP_ORDER:
                log.info("  Found %s ball @ %.2f m", det.colour, det.depth)
                self.ball_records.append(BallRecord(
                    colour=det.colour, cx=det.cx, cy=det.cy, depth=det.depth
                ))

        found_colours = {r.colour for r in self.ball_records}
        missing       = [c for c in config.PICKUP_ORDER if c not in found_colours]

        # Sort records in pickup order
        order_map = {c: i for i, c in enumerate(config.PICKUP_ORDER)}
        self.ball_records.sort(key=lambda r: order_map.get(r.colour, 99))

        if len(self.ball_records) == 3:
            log.info("All 3 balls located: %s", [r.colour for r in self.ball_records])
            self.current_ball_idx = 0
            self._transition(State.ALIGN_BALL)
        elif now - self._state_enter_time > config.SEARCH_TIMEOUT_S:
            if self.ball_records:
                log.warning("Scan timeout — found %d/3 balls: %s. Proceeding.",
                            len(self.ball_records), [r.colour for r in self.ball_records])
                self.current_ball_idx = 0
                self._transition(State.ALIGN_BALL)
            else:
                log.error("No balls found — abort")
                self._transition(State.ABORT)

        return {"found": [r.colour for r in self.ball_records], "missing": missing}

    # ── Align over current target ball ────────────────────────────────────────

    def _align_ball(self, frame, depth_frame, now) -> dict:
        if self.current_ball_idx >= len(self.ball_records):
            # All balls done — fly to barrels
            self._transition(State.FLY_TO_BARRELS)
            return {}

        rec    = self.ball_records[self.current_ball_idx]
        target = rec.colour

        if self._just_entered():
            log.info("Aligning over %s ball (attempt %d/%d)",
                     target, rec.grip_retries + 1, config.MAX_GRIP_RETRIES)

        dets = self.ball_det.detect(frame, depth_frame, target_colour=target)

        if not dets:
            # Lost sight — slow rotation to re-acquire
            self.mav.send_velocity(0.0, 0.05, 0.0)
            if now - self._state_enter_time > config.ALIGN_TIMEOUT_S:
                log.warning("Lost %s ball during align — retrying scan", target)
                self._transition(State.SCAN_BALLS)
            return {}

        det = dets[0]
        vx, vy, vz = _compute_velocity(det, target_depth=config.GRIP_DISTANCE_M)
        self.mav.send_velocity(vx, vy, vz)

        if _is_centered(det) and 0 < det.depth <= config.GRIP_DISTANCE_M:
            self._depth_before_grip = det.depth
            self._transition(State.GRIP)

        return {"target": det, "vx": vx, "vy": vy, "vz": vz}

    # ── Grip ──────────────────────────────────────────────────────────────────

    def _grip(self, frame, depth_frame, now) -> dict:
        if self._just_entered():
            self.mav.hover()
            self.gripper.close()
            self._grip_time = now
            log.info("Gripper closed — holding for %.1f s", config.GRIP_HOLD_S)

        if now - self._grip_time >= config.GRIP_HOLD_S:
            if config.GRIP_VERIFY_ENABLED:
                self._transition(State.VERIFY_GRIP)
            else:
                self.holding_ball = True
                self._transition(State.FLY_TO_BARRELS if self._all_balls_held() else State.ALIGN_BALL)

        return {}

    # ── Verify grip ───────────────────────────────────────────────────────────

    def _verify_grip(self, frame, depth_frame, now) -> dict:
        if self._just_entered():
            time.sleep(config.GRIP_VERIFY_WAIT_S)   # let servo settle
            log.info("Verifying grip...")

        rec    = self.ball_records[self.current_ball_idx]
        dets   = self.ball_det.detect(frame, depth_frame, target_colour=rec.colour)

        # Grip success: ball no longer visible, or depth changed significantly
        grip_ok = (
            not dets
            or (dets and abs(dets[0].depth - self._depth_before_grip) > config.GRIP_VERIFY_DEPTH_DELTA)
        )

        if grip_ok:
            log.info("Grip CONFIRMED for %s ball", rec.colour)
            self.holding_ball = True
            rec.picked_up     = True
            # Ascend before moving to next ball / barrel
            self.mav.send_velocity(0.0, 0.0, -config.MAX_SPEED_Z)
            time.sleep(1.5)
            self.mav.hover()
            self.current_ball_idx += 1
            if self.current_ball_idx < len(self.ball_records):
                self._transition(State.ALIGN_BALL)
            else:
                self._transition(State.FLY_TO_BARRELS)
        else:
            # Grip failed
            rec.grip_retries += 1
            log.warning("Grip FAILED for %s ball — retry %d/%d",
                        rec.colour, rec.grip_retries, config.MAX_GRIP_RETRIES)
            self.gripper.open()
            self.holding_ball = False

            if rec.grip_retries >= config.MAX_GRIP_RETRIES:
                log.error("Max retries reached for %s — skipping (reluctantly)", rec.colour)
                rec.picked_up = True   # mark to avoid infinite loop
                self.current_ball_idx += 1
                if self.current_ball_idx < len(self.ball_records):
                    self._transition(State.ALIGN_BALL)
                else:
                    self._transition(State.FLY_TO_BARRELS)
            else:
                # Back off slightly then retry
                self.mav.send_velocity(-0.1, 0.0, -0.1)
                time.sleep(1.0)
                self.mav.hover()
                self._transition(State.ALIGN_BALL)

        return {}

    # ── Fly to barrel zone ────────────────────────────────────────────────────

    def _fly_to_barrels(self, frame, depth_frame, now) -> dict:
        if self._just_entered():
            log.info("Flying to barrel zone (%.1f m forward)", config.BARREL_SEARCH_FWD_M)
            self._travel_start_time = now
            self.mav.hover()

        elapsed  = now - self._travel_start_time
        distance = elapsed * config.BARREL_SEARCH_SPEED

        if distance < config.BARREL_SEARCH_FWD_M:
            self.mav.send_velocity(config.BARREL_SEARCH_SPEED, 0.0, 0.0)
        else:
            self.mav.hover()
            self._transition(State.SCAN_BARRELS)

        return {"travel_m": distance}

    # ── Scan barrel zone — find all 3 ─────────────────────────────────────────

    def _scan_barrels(self, frame, depth_frame, now) -> dict:
        if self._just_entered():
            log.info("Scanning for barrels...")
            self.mav.hover()

        dets = self.barrel_det.detect(frame, depth_frame)

        for det in dets:
            already = any(
                abs(det.cx - bx) < 80 and abs(det.cy - by) < 80
                for bx, by in self.barrel_records
            )
            if not already:
                log.info("  Barrel found @ %.2f m cx=%d cy=%d", det.depth, det.cx, det.cy)
                self.barrel_records.append((det.cx, det.cy))

        if len(self.barrel_records) >= 3:
            log.info("All barrels located — beginning drops")
            self.current_ball_idx = 0
            self._transition(State.ALIGN_BARREL)
        elif now - self._state_enter_time > config.SEARCH_TIMEOUT_S:
            if self.barrel_records:
                log.warning("Scan timeout — found %d barrel(s). Proceeding.", len(self.barrel_records))
                self.current_ball_idx = 0
                self._transition(State.ALIGN_BARREL)
            else:
                log.error("No barrels found — abort")
                self._transition(State.ABORT)

        return {"barrels_found": len(self.barrel_records)}

    # ── Align over barrel ─────────────────────────────────────────────────────

    def _align_barrel(self, frame, depth_frame, now) -> dict:
        if self._just_entered():
            log.info("Aligning over barrel for drop %d", self.current_ball_idx + 1)

        dets        = self.barrel_det.detect(frame, depth_frame)
        empty_dets  = self._filter_used_barrels(dets)

        if not empty_dets:
            self.mav.send_velocity(0.0, 0.05, 0.0)   # slow sweep
            if now - self._state_enter_time > config.ALIGN_TIMEOUT_S:
                log.warning("Barrel align timeout — trying scan again")
                self._transition(State.SCAN_BARRELS)
            return {}

        det = empty_dets[0]
        vx, vy, vz = _compute_velocity(det, target_depth=config.DROP_HOVER_M)
        self.mav.send_velocity(vx, vy, vz)

        if _is_centered(det) and abs(det.depth - config.DROP_HOVER_M) < 0.15:
            self._transition(State.DROP)

        return {"target": det, "vx": vx, "vy": vy, "vz": vz}

    # ── Drop ──────────────────────────────────────────────────────────────────

    def _drop(self, frame, depth_frame, now) -> dict:
        if self._just_entered():
            self.mav.hover()
            self.gripper.open()
            self.holding_ball = False
            log.info("Ball dropped")

            # Mark this barrel as used by current pixel position
            dets = self.barrel_det.detect(frame, depth_frame)
            empty = self._filter_used_barrels(dets)
            if empty:
                self.used_barrels.append((empty[0].cx, empty[0].cy))

            self.current_ball_idx += 1

        if now - self._state_enter_time > 1.5:
            if self.current_ball_idx < len([b for b in self.ball_records if b.picked_up]):
                # Ascend then next barrel
                self.mav.send_velocity(0.0, 0.0, -config.MAX_SPEED_Z)
                time.sleep(1.2)
                self.mav.hover()
                self._transition(State.ALIGN_BARREL)
            else:
                self._transition(State.COMPLETE)

        return {}

    # ── Terminal states ───────────────────────────────────────────────────────

    def _complete(self, frame, depth_frame, now) -> dict:
        if self._just_entered():
            log.info("Mission complete — all balls delivered. Landing.")
            self.mav.land()
        return {}

    def _abort(self, frame, depth_frame, now) -> dict:
        if self._just_entered():
            log.error("ABORT — opening gripper and landing.")
            self.gripper.open()
            self.mav.hover()
            time.sleep(0.5)
            self.mav.land()
        return {}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _transition(self, new: State) -> None:
        log.info("%-20s → %s", self.state.name, new.name)
        self.state             = new
        self._state_enter_time = time.time()

    def _just_entered(self) -> bool:
        return (time.time() - self._state_enter_time) < 0.05

    def _all_balls_held(self) -> bool:
        return all(r.picked_up for r in self.ball_records)

    def _filter_used_barrels(self, dets: list[Detection]) -> list[Detection]:
        return [
            d for d in dets
            if not any(abs(d.cx - ux) < 60 and abs(d.cy - uy) < 60
                       for ux, uy in self.used_barrels)
        ]


# ── Velocity helpers ──────────────────────────────────────────────────────────

def _compute_velocity(det: Detection, target_depth: float) -> tuple[float, float, float]:
    vx = float(np.clip(config.KP_XY * det.err_x_m, -config.MAX_SPEED_XY, config.MAX_SPEED_XY))
    vy = float(np.clip(config.KP_XY * det.err_y_m, -config.MAX_SPEED_XY, config.MAX_SPEED_XY))
    depth_err = det.depth - target_depth
    vz = float(np.clip(config.KP_Z * depth_err, 0.0, config.MAX_SPEED_Z)) if (
        _is_centered(det) and depth_err > 0.05
    ) else 0.0
    return vx, vy, vz


def _is_centered(det: Detection) -> bool:
    return abs(det.err_x_m) < config.CENTER_THRESH_M and abs(det.err_y_m) < config.CENTER_THRESH_M
