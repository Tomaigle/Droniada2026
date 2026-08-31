from __future__ import annotations

import cv2
import numpy as np

from generator.regulation import GRID, COLORS, CORNER_MARKER_CELL, CORNER_MARKER_BGR, cell_bounds_px, hex_to_bgr


def render_panel(
    cards: list[tuple[int, int, str]],
    size_px: tuple[int, int] = (1000, 500),
    show_grid: bool = False,
    corner_marker: bool = True,
) -> np.ndarray:
    """Render one top-down panel with black background and exactly one filled cell per card.
    
    If corner_marker=True, fills cell (1,1) with white first (the regulation corner marker).
    Cards are drawn after, but no card can be at (1,1) so there is no overlap.
    """
    width_px, height_px = size_px
    img = np.zeros((height_px, width_px, 3), dtype=np.uint8)
    cell_w = width_px / GRID[0]
    cell_h = height_px / GRID[1]

    # Draw corner marker first (if enabled)
    if corner_marker:
        marker_x, marker_y = CORNER_MARKER_CELL
        x0, y_top, x1, y1 = cell_bounds_px(marker_x, marker_y, size_px)
        cv2.rectangle(img, (x0, y_top), (x1, y1), CORNER_MARKER_BGR, thickness=-1)

    for x, y, color_name in cards:
        if not (1 <= x <= GRID[0]) or not (1 <= y <= GRID[1]):
            raise ValueError(f"Card coordinates out of range: ({x}, {y})")
        x0, y_top, x1, y1 = cell_bounds_px(x, y, size_px)
        color = hex_to_bgr(dict(COLORS)[color_name])
        cv2.rectangle(img, (x0, y_top), (x1, y1), color, thickness=-1)

    if show_grid:
        line_color = (80, 80, 80)
        for ix in range(1, GRID[0]):
            x = int(round(ix * cell_w))
            cv2.line(img, (x, 0), (x, height_px), line_color, 1)
        for iy in range(1, GRID[1]):
            y = int(round(iy * cell_h))
            cv2.line(img, (0, y), (width_px, y), line_color, 1)

    return img
