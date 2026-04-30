"""
mission.py — Last Mile Logistics BASIC stage state machine.

Mission flow per ball (repeated 3×: blue → red → yellow):
  SEARCH_BALL  → fly slow search pattern until target ball found
  ALIGN_BALL   → centre drone over ball, descend to grip distance
  GRIP         → close gripper, ascend to pickup hover height
  SEARCH_BARREL→ fly toward barrel zone until barrel found
  ALIGN_BARREL → centre drone over empty barrel
  DROP         → open gripper, ascend, mark barrel used

After all 3 balls: RTL / land.
"""

import time
import logging
import numpy as np
from enum import Enum, auto
from typing import Optional

from mavlink_controller import MAVLinkController
from detector import Detection, BallDetector, BarrelDetector, RealSenseCamera
import config

log = logging.getLogger(__name__)


class State(Enum):
    IDLE          = auto()
    TAKEOFF       = auto()
    SEARCH_BALL   = auto()
    ALIGN_BALL    = auto()
    GRIP          = auto()
    SEARCH_BARREL = auto()
    ALIGN_BARREL  = auto()
    DROP          = auto()
    COMPLETE      = auto()
    ABORT         = auto()


class MissionController:
    def __init__(
        self,
        mav: MAVLinkController,
        camera: RealSenseCamera,
        ball_detector: BallDetector,
        barrel_detector: BarrelDetector,
    ):
        self.mav = mav
        self.cam = camera
        self.ball_det = ball_detector
        self.barrel_det = barrel_detector

        self.state = State.IDLE
        self.pickup_queue = list(config.PICKUP_ORDER)   # ["blue","red","yellow"]
        self.current_colour: Optional[str] = None
        self.used_barrels: list[tuple[int,int]] = []    # (cx,cy) of already-used barrels

        self._state_enter_time = 0.0
        self._mission_start    = 0.0
        self._grip_time        = 0.0
        self.holding_ball      = False

    # ── Public entry ──────────────────────────────────────────────────────────

    def run_step(self, frame, depth_frame) -> dict:
        """
        Call once per camera frame.
        Returns telemetry dict for overlay / logging.
        """
        now = time.time()

        # Global mission timeout guard
        if self._mission_start and (now - self._mission_start) > config.MISSION_TIMEOUT_S:
            log.warning("Mission timeout — landing")
            self._transition(State.ABORT)

        handler = {
            State.IDLE:           self._idle,
            State.TAKEOFF:        self._takeoff,
            State.SEARCH_BALL:    self._search_ball,
            State.ALIGN_BALL:     self._align_ball,
            State.GRIP:           self._grip,
            State.SEARCH_BARREL:  self._search_barrel,
            State.ALIGN_BARREL:   self._align_barrel,
            State.DROP:           self._drop,
            State.COMPLETE:       self._complete,
            State.ABORT:          self._abort,
        }.get(self.state, self._idle)

        telemetry = handler(frame, depth_frame, now)
        telemetry["state"]   = self.state.name
        telemetry["queue"]   = list(self.pickup_queue)
        telemetry["holding"] = self.holding_ball
        return telemetry

    def start(self) -> None:
        self._mission_start = time.time()
        self._transition(State.TAKEOFF)

    # ── State handlers ────────────────────────────────────────────────────────

    def _idle(self, frame, depth_frame, now) -> dict:
        self.mav.hover()
        return {}

    def _takeoff(self, frame, depth_frame, now) -> dict:
        if self._just_entered():
            self.mav.set_mode("GUIDED")
            time.sleep(0.5)
            self.mav.arm()
            time.sleep(2.0)
            self.mav.takeoff(config.PICKUP_HOVER_M)
            log.info("Takeoff initiated")
        # Wait ~5 s then start looking for first ball
        if now - self._state_enter_time > 6.0:
            self.current_colour = self.pickup_queue[0]
            self._transition(State.SEARCH_BALL)
        return {}

    def _search_ball(self, frame, depth_frame, now) -> dict:
        if self._just_entered():
            log.info("Searching for %s ball", self.current_colour)

        detections = self.ball_det.detect(frame, depth_frame, target_colour=self.current_colour)

        if detections:
            log.info("Ball found: %s @ %.2f m", detections[0].colour, detections[0].depth)
            self._transition(State.ALIGN_BALL)
            return {"target": detections[0]}

        # Simple search: slow forward creep — upgrade to pattern if needed
        self.mav.send_velocity(0.2, 0.0, 0.0)

        if now - self._state_enter_time > config.SEARCH_TIMEOUT_S:
            log.warning("Ball %s not found in time — skipping", self.current_colour)
            self._skip_ball()

        return {}

    def _align_ball(self, frame, depth_frame, now) -> dict:
        detections = self.ball_det.detect(frame, depth_frame, target_colour=self.current_colour)

        if not detections:
            # Lost ball — go back to search
            self.mav.hover()
            self._transition(State.SEARCH_BALL)
            return {}

        det = detections[0]
        vx, vy, vz = _compute_approach_velocity(det)
        self.mav.send_velocity(vx, vy, vz)

        centered = _is_centered(det)
        at_grip  = 0 < det.depth < config.GRIP_DISTANCE_M

        if centered and at_grip:
            self._transition(State.GRIP)

        if now - self._state_enter_time > config.ALIGN_TIMEOUT_S:
            log.warning("Align timeout on %s — skipping", self.current_colour)
            self._skip_ball()

        return {"target": det, "vx": vx, "vy": vy, "vz": vz}

    def _grip(self, frame, depth_frame, now) -> dict:
        if self._just_entered():
            self.mav.hover()
            self.mav.gripper_close()
            self._grip_time = now
            log.info("Gripping %s ball", self.current_colour)

        if now - self._grip_time >= config.GRIP_HOLD_S:
            self.holding_ball = True
            # Ascend back to pickup hover height before searching barrel
            self.mav.send_velocity(0.0, 0.0, -config.MAX_SPEED_Z)   # up in NED = negative Z
            time.sleep(1.5)
            self.mav.hover()
            self._transition(State.SEARCH_BARREL)

        return {}

    def _search_barrel(self, frame, depth_frame, now) -> dict:
        if self._just_entered():
            log.info("Searching for empty barrel")

        detections = self.barrel_det.detect(frame, depth_frame)
        empty_barrels = self._filter_used_barrels(detections)

        if empty_barrels:
            log.info("Barrel found @ %.2f m", empty_barrels[0].depth)
            self._transition(State.ALIGN_BARREL)
            return {"target": empty_barrels[0]}

        self.mav.send_velocity(0.2, 0.0, 0.0)

        if now - self._state_enter_time > config.SEARCH_TIMEOUT_S:
            log.warning("Barrel not found — aborting drop for this ball")
            self._drop_ball_without_barrel()

        return {}

    def _align_barrel(self, frame, depth_frame, now) -> dict:
        detections = self.barrel_det.detect(frame, depth_frame)
        empty_barrels = self._filter_used_barrels(detections)

        if not empty_barrels:
            self.mav.hover()
            self._transition(State.SEARCH_BARREL)
            return {}

        det = empty_barrels[0]
        vx, vy, vz = _compute_approach_velocity(det, target_depth=config.DROP_HOVER_M)
        self.mav.send_velocity(vx, vy, vz)

        centered = _is_centered(det)
        at_drop  = abs(det.depth - config.DROP_HOVER_M) < 0.15

        if centered and at_drop:
            self._transition(State.DROP)

        return {"target": det, "vx": vx, "vy": vy, "vz": vz}

    def _drop(self, frame, depth_frame, now) -> dict:
        if self._just_entered():
            self.mav.hover()
            self.mav.gripper_open()
            self.holding_ball = False
            log.info("Ball dropped into barrel")

            # Mark barrel as used
            detections = self.barrel_det.detect(frame, depth_frame)
            empty = self._filter_used_barrels(detections)
            if empty:
                self.used_barrels.append((empty[0].cx, empty[0].cy))

            # Consume ball from queue
            self.pickup_queue.pop(0)

        # Wait a moment then decide next step
        if now - self._state_enter_time > 1.5:
            if self.pickup_queue:
                self.current_colour = self.pickup_queue[0]
                # Ascend before searching next ball
                self.mav.send_velocity(0.0, 0.0, -config.MAX_SPEED_Z)
                time.sleep(1.5)
                self.mav.hover()
                self._transition(State.SEARCH_BALL)
            else:
                self._transition(State.COMPLETE)

        return {}

    def _complete(self, frame, depth_frame, now) -> dict:
        if self._just_entered():
            log.info("Mission complete — all 3 balls delivered. Landing.")
            self.mav.land()
        return {}

    def _abort(self, frame, depth_frame, now) -> dict:
        if self._just_entered():
            log.error("Mission aborted — landing now.")
            self.mav.gripper_open()
            self.mav.land()
        return {}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _transition(self, new_state: State) -> None:
        log.info("State: %s → %s", self.state.name, new_state.name)
        self.state = new_state
        self._state_enter_time = time.time()

    def _just_entered(self) -> bool:
        return (time.time() - self._state_enter_time) < 0.05

    def _skip_ball(self) -> None:
        """Skip current ball; still attempt remaining ones."""
        if self.pickup_queue:
            self.pickup_queue.pop(0)
        if self.pickup_queue:
            self.current_colour = self.pickup_queue[0]
            self._transition(State.SEARCH_BALL)
        else:
            self._transition(State.COMPLETE)

    def _drop_ball_without_barrel(self) -> None:
        """Emergency: drop ball and move on."""
        self.mav.gripper_open()
        self.holding_ball = False
        self._skip_ball()

    def _filter_used_barrels(self, detections: list[Detection]) -> list[Detection]:
        """Remove barrels that have already received a ball (by proximity)."""
        result = []
        for det in detections:
            used = any(
                abs(det.cx - ux) < 60 and abs(det.cy - uy) < 60
                for ux, uy in self.used_barrels
            )
            if not used:
                result.append(det)
        return result


# ── Velocity helpers (shared by align states) ─────────────────────────────────

def _compute_approach_velocity(
    det: Detection,
    target_depth: float = config.GRIP_DISTANCE_M,
) -> tuple[float, float, float]:
    """P-controller: drive XY error to zero, Z toward target_depth."""
    vx = float(np.clip(config.KP_XY * det.err_x_m, -config.MAX_SPEED_XY, config.MAX_SPEED_XY))
    vy = float(np.clip(config.KP_XY * det.err_y_m, -config.MAX_SPEED_XY, config.MAX_SPEED_XY))

    centered = _is_centered(det)
    depth_err = det.depth - target_depth
    if centered and depth_err > 0.05:
        vz = float(np.clip(config.KP_Z * depth_err, 0.0, config.MAX_SPEED_Z))
    else:
        vz = 0.0

    return vx, vy, vz


def _is_centered(det: Detection) -> bool:
    return abs(det.err_x_m) < config.CENTER_THRESH_M and abs(det.err_y_m) < config.CENTER_THRESH_M
