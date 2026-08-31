from __future__ import annotations

PANEL_SIZE_CM = (200, 100)
GRID = (10, 10)
CELL_CM = (20, 10)

COLORS = [
    ("czerwona", "#FF0000"),
    ("pomarańczowa", "#FF7A00"),
    ("żółta", "#FFE000"),
    ("zielona", "#00A000"),
    ("niebieska", "#0000FF"),
    ("fioletowa", "#B000C0"),
]


def hex_to_bgr(hex: str) -> tuple[int, int, int]:
    """Convert a 6-digit hex color to OpenCV BGR tuple."""
    value = hex.strip().lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if len(value) != 6:
        raise ValueError(f"Invalid hex color: {hex!r}")
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return b, g, r


def cell_bounds_cm(x: int, y: int) -> tuple[float, float, float, float]:
    """Return the 2D bounds of one grid cell in cm relative to the bottom-left panel corner."""
    if not (1 <= x <= GRID[0]) or not (1 <= y <= GRID[1]):
        raise ValueError(f"Cell coordinates out of range: ({x}, {y})")
    x0 = (x - 1) * CELL_CM[0]
    y0 = (y - 1) * CELL_CM[1]
    x1 = x0 + CELL_CM[0]
    y1 = y0 + CELL_CM[1]
    return x0, y0, x1, y1


def cell_bounds_px(x: int, y: int, size_px: tuple[int, int]) -> tuple[int, int, int, int]:
    """Return the image-space bounds for a grid cell using the Y-flip convention used by the detector."""
    if not (1 <= x <= GRID[0]) or not (1 <= y <= GRID[1]):
        raise ValueError(f"Cell coordinates out of range: ({x}, {y})")
    width_px, height_px = size_px
    cell_w = width_px / GRID[0]
    cell_h = height_px / GRID[1]
    x0 = (x - 1) * cell_w
    y_top = (GRID[1] - y) * cell_h
    x1 = x0 + cell_w
    y1 = y_top + cell_h
    return int(x0), int(y_top), int(x1), int(y1)
