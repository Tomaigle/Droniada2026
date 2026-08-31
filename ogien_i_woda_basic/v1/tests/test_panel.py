import numpy as np

from generator.panel import render_panel


def test_render_panel_shape_and_dtype():
    img = render_panel([(1, 1, "czerwona")], size_px=(1000, 500), show_grid=False)
    assert img.shape == (500, 1000, 3)
    assert img.dtype == np.uint8


def test_render_panel_red_bottom_left_green_top_left():
    # (1,1) = bottom-left cell -> image rows 450..500
    img = render_panel([(1, 1, "czerwona")], size_px=(1000, 500), show_grid=False)
    b, g, r = img[490, 5]
    assert r > 200 and g < 25 and b < 25

    # (1,10) = top-left cell -> image rows 0..50
    img = render_panel([(1, 10, "zielona")], size_px=(1000, 500), show_grid=False)
    b, g, r = img[10, 5]
    assert g > 120 and r < 25 and b < 25  # zielona is #00A000 -> G=160


def test_render_panel_cards_do_not_bleed_and_empty_stays_black():
    img = render_panel([(1, 1, "czerwona"), (2, 1, "niebieska")], size_px=(1000, 500), show_grid=False)

    # cell (1,1) is red
    b, g, r = img[490, 50]
    assert r > 200 and g < 25 and b < 25

    # cell (2,1) is blue, not red -> colors did not bleed
    b, g, r = img[490, 150]
    assert b > 200 and g < 25 and r < 25

    # cell (5,5) has no card -> still black
    assert img[275, 450].tolist() == [0, 0, 0]


def test_corner_marker_default_true():
    """Cell (1,1) is white when corner_marker=True (default)."""
    img = render_panel([], size_px=(1000, 500), corner_marker=True)
    # Center of cell (1,1): row ~490, col ~50
    b, g, r = img[490, 50]
    assert b == 255 and g == 255 and r == 255, f"Expected white (255,255,255), got ({b},{g},{r})"


def test_corner_marker_false():
    """Cell (1,1) is black when corner_marker=False."""
    img = render_panel([], size_px=(1000, 500), corner_marker=False)
    # Center of cell (1,1): row ~490, col ~50
    b, g, r = img[490, 50]
    assert b == 0 and g == 0 and r == 0, f"Expected black (0,0,0), got ({b},{g},{r})"
