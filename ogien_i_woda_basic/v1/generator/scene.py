from __future__ import annotations

import math

import cv2
import numpy as np

from generator.panel import render_panel
from generator.regulation import cell_bounds_px
from generator.sampler import PanelSpec

PANEL_SIZE_PX = (1000, 500)  # (width, height) of the flat rendered panel
PANEL_X_POSITIONS = [0.20, 0.50, 0.80]  # panel centre X as fraction of frame width
PANEL_Y_POSITION = 0.60  # panel centre Y as fraction of frame height
PANEL_ON_SCREEN_FRAC = 0.42  # shrink each panel on screen so 3 banners fit the 50x50 m
#                              field WITHOUT overlapping — overlapping black panels merge
#                              into one blob and the detector can only find 1 panel.
BACKGROUND_BGR = (120, 120, 120)  # non-black ground: the panels are black banners on a
#                                   field, so the scene MUST have contrast or the
#                                   detector segments the whole frame as one black blob.


def rotate_quad(quad: np.ndarray, center: tuple[float, float], angle_deg: float) -> np.ndarray:
    """Rotate a 4-point quad around a centre by angle_deg (in-plane, degrees)."""
    if angle_deg == 0.0:
        return quad
    angle = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    cx, cy = center
    rotated = []
    for x, y in quad:
        dx, dy = x - cx, y - cy
        rotated.append([cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a])
    return np.array(rotated, dtype=np.float32)


def build_panel_projection(
    panel_center: tuple[float, float],
    camera_yaw_deg: float,
    camera_pitch_deg: float,
    panel_rotation_deg: float,
    panel_width_px: int,
    panel_height_px: int,
    distance: float,
) -> np.ndarray:
    """Project the flat panel rectangle to an image-space quad, given camera + panel orientation.

    Ported from images/generate_panel_images.py (the camera_yaw/camera_pitch variant).
    Panel ground-tilt is intentionally dropped: the regulation only defines in-plane
    orientation 0/45/90, so panel_rotation_deg carries that.
    """
    cx, cy = panel_center
    scale = max(0.3, min(1.2, 4000.0 / max(1.0, distance))) * PANEL_ON_SCREEN_FRAC
    w = panel_width_px * scale
    h = panel_height_px * scale

    yaw_skew = math.tan(math.radians(camera_yaw_deg)) * w * 0.15
    pitch_scale = max(0.25, 1.0 - abs(camera_pitch_deg) / 90.0 * 0.6)
    top_h = h * pitch_scale

    left_x, right_x = cx - w * 0.5, cx + w * 0.5
    bottom_y, top_y = cy + h * 0.5, cy - top_h * 0.5

    quad = np.array(
        [[left_x, bottom_y], [right_x, bottom_y], [right_x + yaw_skew, top_y], [left_x + yaw_skew, top_y]],
        dtype=np.float32,
    )
    return rotate_quad(quad, (cx, cy), panel_rotation_deg)


def panel_homography(
    spec: PanelSpec,
    panel_index: int,
    total_panels: int,
    width: int,
    height: int,
    distance: float,
    yaw: float,
    pitch: float,
) -> np.ndarray | None:
    """3x3 homography mapping flat-panel pixels (0..W, 0..H, origin top-left) to the frame."""
    frac = PANEL_X_POSITIONS[panel_index] if total_panels > 1 else PANEL_X_POSITIONS[1]
    center = (width * frac, height * PANEL_Y_POSITION)
    pw, ph = PANEL_SIZE_PX
    dst = build_panel_projection(center, yaw, pitch, spec.orientation_deg, pw, ph, distance)
    if np.any(np.isnan(dst)):
        return None
    # build_panel_projection's dst is ordered screen BL, BR, TR, TL. render_panel already
    # Y-flips (grid y=1 at the flat-image BOTTOM), so pair the flat BOTTOM edge (y=ph) with
    # the screen BOTTOM. Pairing (0,0) here instead double-flips the panel upside down.
    src = np.array([[0.0, ph], [pw, ph], [pw, 0.0], [0.0, 0.0]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def _project(h: np.ndarray, pts: list[tuple[float, float]]) -> list[list[float]]:
    arr = np.array([pts], dtype=np.float32)
    out = cv2.perspectiveTransform(arr, h)[0]
    return [p.tolist() for p in out]


def project_cell_center_px(h: np.ndarray, x: int, y: int) -> list[float]:
    """Image-space centre of grid cell (x, y) for a panel with homography h."""
    x0, y0, x1, y1 = cell_bounds_px(x, y, PANEL_SIZE_PX)
    return _project(h, [((x0 + x1) / 2.0, (y0 + y1) / 2.0)])[0]


def panel_corner_quad_px(h: np.ndarray) -> list[list[float]]:
    """The 4 panel corners in image space, order: (1,1) BL, (10,1) BR, (10,10) TR, (1,10) TL."""
    pw, ph = PANEL_SIZE_PX
    # grid BL=(0,H) BR=(W,H) TR=(W,0) TL=(0,0) in flat-panel pixels (origin top-left)
    return _project(h, [(0.0, ph), (pw, ph), (pw, 0.0), (0.0, 0.0)])


def compose_scene(
    specs: list[PanelSpec],
    width: int = 1920,
    height: int = 1080,
    distance: float = 4000.0,
    yaw: float = 0.0,
    pitch: float = 25.0,
    show_grid: bool = False,
) -> tuple[np.ndarray, list[list[list[float]]]]:
    """Compose the multi-panel scene with perspective projection. Returns (image, quads_px)."""
    output = np.full((height, width, 3), BACKGROUND_BGR, dtype=np.uint8)
    pw, ph = PANEL_SIZE_PX
    quads_px: list[list[list[float]]] = []

    for idx, spec in enumerate(sorted(specs, key=lambda s: s.panel_id)):
        h = panel_homography(spec, idx, len(specs), width, height, distance, yaw, pitch)
        if h is None:
            continue

        cards = [(c.x, c.y, c.color) for c in spec.cards]
        panel_img = render_panel(cards, size_px=PANEL_SIZE_PX, show_grid=show_grid)
        warped = cv2.warpPerspective(panel_img, h, (width, height), flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(
            np.full((ph, pw), 255, dtype=np.uint8), h, (width, height), flags=cv2.INTER_NEAREST
        ).astype(bool)
        output[mask] = warped[mask]

        quads_px.append(panel_corner_quad_px(h))

    return output, quads_px
