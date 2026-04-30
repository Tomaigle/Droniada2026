"""
overlay.py — OpenCV HUD overlay for live debug feed.
"""

import cv2
import numpy as np
from detector import Detection
import config

_C = {
    "blue":    (200,  80,   0),
    "red":     (  0,  40, 220),
    "yellow":  (  0, 210, 230),
    "barrel":  (150, 150,   0),
    "unknown": (120, 120, 120),
}
_WHITE  = (255, 255, 255)
_GREEN  = (  0, 220,  60)
_ORANGE = (  0, 165, 255)
_GRAY   = (160, 160, 160)
_FONT   = cv2.FONT_HERSHEY_SIMPLEX


def draw_frame(
    frame:       np.ndarray,
    state_name:  str,
    telemetry:   dict,
    detections:  list[Detection],
    holding:     bool,
    queue:       list[str],
    vx: float = 0.0,
    vy: float = 0.0,
    vz: float = 0.0,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    fc = (w // 2, h // 2)

    # Frame centre crosshair
    cv2.line(out, (fc[0]-20, fc[1]), (fc[0]+20, fc[1]), _WHITE, 1)
    cv2.line(out, (fc[0], fc[1]-20), (fc[0], fc[1]+20), _WHITE, 1)

    target: Detection | None = telemetry.get("target")

    for det in detections:
        col = _C.get(det.colour, _GRAY)
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 1)
        label = f"{det.colour} {det.depth:.2f}m"
        cv2.putText(out, label, (x1, y1-6), _FONT, 0.45, col, 1, cv2.LINE_AA)

        if target and det.cx == target.cx and det.cy == target.cy:
            cv2.circle(out, (det.cx, det.cy), 12, col, 2)
            cv2.arrowedLine(out, fc, (det.cx, det.cy), col, 2, tipLength=0.2)
            err_txt = f"err ({det.err_x_m:+.3f}, {det.err_y_m:+.3f}) m"
            cv2.putText(out, err_txt, (x1, y2+16), _FONT, 0.45, col, 1, cv2.LINE_AA)

    # Status panel top-left
    lines = [
        (f"STATE : {state_name}",                        _GREEN if "COMPLETE" in state_name else _ORANGE),
        (f"VEL   vx={vx:+.2f} vy={vy:+.2f} vz={vz:+.2f}", _WHITE),
        (f"GRIP  {'HOLDING' if holding else 'OPEN'}",    _GREEN if holding else _GRAY),
        (f"QUEUE {' → '.join(queue) if queue else 'done'}", _WHITE),
    ]
    y = 26
    for txt, col in lines:
        cv2.putText(out, txt, (10, y), _FONT, 0.58, col, 2, cv2.LINE_AA)
        y += 24

    # Tilt indicator bottom-right
    tilt_txt = f"TILT {config.CAMERA_TILT_DEG:.0f}deg"
    cv2.putText(out, tilt_txt, (w-140, h-10), _FONT, 0.45, _GRAY, 1, cv2.LINE_AA)

    return out
