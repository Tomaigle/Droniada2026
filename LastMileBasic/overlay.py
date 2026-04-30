"""
overlay.py — OpenCV HUD overlay for live debug feed.

Draws:
  - Detection bounding boxes + depth labels
  - Error arrows from frame centre to target
  - State machine status
  - Velocity commands
  - Gripper status
  - Mission queue
"""

import cv2
import numpy as np
from detector import Detection
import config

# Colour palette (BGR)
_COLOURS = {
    "blue":    (200, 80,  0),
    "red":     (0,   40, 220),
    "yellow":  (0,   210, 230),
    "barrel":  (150, 150, 0),
    "unknown": (120, 120, 120),
}
_WHITE  = (255, 255, 255)
_GREEN  = (0,   220, 60)
_ORANGE = (0,   165, 255)
_RED    = (0,   40,  220)
_GRAY   = (160, 160, 160)

_FONT    = cv2.FONT_HERSHEY_SIMPLEX
_SMALL   = 0.50
_MEDIUM  = 0.62
_THICK   = 2
_THIN    = 1


def draw_frame(
    frame: np.ndarray,
    state_name: str,
    telemetry: dict,
    detections: list[Detection],
    holding: bool,
    queue: list[str],
    vx: float = 0.0,
    vy: float = 0.0,
    vz: float = 0.0,
) -> np.ndarray:
    """Return annotated copy of frame."""
    out = frame.copy()
    h, w = out.shape[:2]
    fc = (w // 2, h // 2)

    # Frame centre crosshair
    cv2.line(out, (fc[0]-20, fc[1]), (fc[0]+20, fc[1]), _WHITE, _THIN)
    cv2.line(out, (fc[0], fc[1]-20), (fc[0], fc[1]+20), _WHITE, _THIN)

    # Detections
    target: Detection | None = telemetry.get("target")
    for det in detections:
        col = _COLOURS.get(det.colour, _GRAY)
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), col, _THIN)
        label = f"{det.colour} {det.depth:.2f}m {det.conf:.0%}"
        cv2.putText(out, label, (x1, y1 - 6), _FONT, _SMALL, col, _THIN, cv2.LINE_AA)

        if target and det.cx == target.cx and det.cy == target.cy:
            # Highlight active target
            cv2.circle(out, (det.cx, det.cy), 12, col, _THICK)
            cv2.arrowedLine(out, fc, (det.cx, det.cy), col, _THICK, tipLength=0.2)

    # ── Top-left status panel ──────────────────────────────────────────────
    lines = [
        (f"STATE: {state_name}", _GREEN if "COMPLETE" in state_name else _ORANGE),
        (f"VEL  vx={vx:+.2f} vy={vy:+.2f} vz={vz:+.2f} m/s", _WHITE),
        (f"GRIP: {'HOLDING' if holding else 'OPEN'}", _GREEN if holding else _GRAY),
        (f"QUEUE: {' → '.join(queue) if queue else 'done'}", _WHITE),
    ]
    y_cursor = 28
    for text, colour in lines:
        cv2.putText(out, text, (10, y_cursor), _FONT, _MEDIUM, colour, _THICK, cv2.LINE_AA)
        y_cursor += 26

    # ── Target depth / error (bottom-left) ────────────────────────────────
    if target:
        info = (
            f"depth={target.depth:.2f}m  "
            f"err=({target.err_x_m:+.3f}, {target.err_y_m:+.3f})m"
        )
        cv2.putText(out, info, (10, h - 12), _FONT, _SMALL, _WHITE, _THIN, cv2.LINE_AA)

    return out
