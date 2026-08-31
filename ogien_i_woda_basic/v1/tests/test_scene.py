import numpy as np

from generator.sampler import Card, PanelSpec, sample_scene
from generator.scene import (
    compose_scene,
    panel_corner_quad_px,
    panel_homography,
    project_cell_center_px,
)


def test_compose_scene_smoke():
    specs = sample_scene(panel_count=3, seed=42)
    img, quads = compose_scene(specs, width=1920, height=1080, distance=4000, yaw=0.0, pitch=25.0)

    assert img.shape == (1080, 1920, 3)
    assert img.dtype == np.uint8
    assert np.any(img > 0)  # something was drawn

    assert len(quads) == 3
    for quad in quads:
        assert len(quad) == 4
        for x, y in quad:
            assert np.isfinite(x) and np.isfinite(y)
            # panels may clip the frame edge; must still be within one frame of it
            assert -1920 <= x <= 3840
            assert -1080 <= y <= 2160


def test_quad_corner_order_bl_br_tr_tl():
    spec = PanelSpec(panel_id=1, orientation_deg=0, cards=[])  # force 0deg for a stable ordering check
    h = panel_homography(spec, 0, 1, 1920, 1080, 4000.0, 0.0, 25.0)
    bl, br, tr, tl = panel_corner_quad_px(h)
    assert bl[0] < br[0]          # bottom-left is left of bottom-right
    assert tl[0] < tr[0]          # top-left is left of top-right
    assert bl[1] > tl[1]          # bottom edge is lower on screen (larger y) than top edge
    assert br[1] > tr[1]


def test_card_colors_land_at_projected_cells():
    """The panel is not double-flipped: a card's colour appears at its projected cell centre."""
    spec = PanelSpec(
        panel_id=1,
        orientation_deg=0,
        cards=[Card(1, 1, "czerwona"), Card(10, 10, "niebieska"), Card(1, 10, "zielona")],
    )
    img, _ = compose_scene([spec], yaw=0.0, pitch=25.0)
    h = panel_homography(spec, 0, 1, 1920, 1080, 4000.0, 0.0, 25.0)

    expected = {
        (1, 1): (0, 0, 255),      # czerwona -> BGR red, bottom-left
        (10, 10): (255, 0, 0),    # niebieska -> BGR blue, top-right
        (1, 10): (0, 160, 0),     # zielona (#00A000) -> BGR, top-left
    }
    for (x, y), (eb, eg, er) in expected.items():
        cx, cy = project_cell_center_px(h, x, y)
        b, g, r = (int(v) for v in img[int(cy), int(cx)])
        assert abs(b - eb) < 40 and abs(g - eg) < 40 and abs(r - er) < 40, f"cell {(x, y)} -> ({b},{g},{r})"
