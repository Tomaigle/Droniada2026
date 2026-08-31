import sys
from pathlib import Path

V1_DIR = Path(__file__).resolve().parents[1]
if str(V1_DIR) not in sys.path:
    sys.path.insert(0, str(V1_DIR))

import main  # noqa: E402
from generator.regulation import COLORS, cell_bounds_cm, cell_bounds_px  # noqa: E402


def test_colors_match_detector_keys():
    assert set(dict(COLORS)) == set(main.COLOR_RANGES)
    assert len(COLORS) == 6


def test_cell_bounds_cm():
    assert cell_bounds_cm(1, 1) == (0.0, 0.0, 20.0, 10.0)
    assert cell_bounds_cm(10, 10) == (180.0, 90.0, 200.0, 100.0)
    assert cell_bounds_cm(1, 10)[1] == 90.0


def test_cell_bounds_px():
    x0, y0, x1, y1 = cell_bounds_px(1, 1, (1000, 500))
    assert x0 == 0 and y1 == 500
    x0, y0, x1, y1 = cell_bounds_px(1, 10, (1000, 500))
    assert y0 == 0


def test_invalid_bounds_raise():
    try:
        cell_bounds_cm(0, 1)
        assert False, "Expected ValueError"
    except ValueError:
        pass
