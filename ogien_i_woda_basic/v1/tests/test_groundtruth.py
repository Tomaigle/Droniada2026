import json

from generator.groundtruth import Camera, write_scene_json
from generator.sampler import sample_scene


def test_write_scene_json_schema(tmp_path):
    specs = sample_scene(panel_count=3, seed=42)
    cam = Camera(distance_mm=4000.0, yaw_deg=10.0, pitch_deg=25.0)
    out = tmp_path / "scene_001.json"
    write_scene_json(out, "scene_001.png", (1920, 1080), cam, 42, specs)

    data = json.loads(out.read_text(encoding="utf-8"))
    assert set(data) == {"image", "image_size", "camera", "seed", "panels"}
    assert data["image"] == "scene_001.png"
    assert data["image_size"] == [1920, 1080]
    assert data["camera"] == {"distance_mm": 4000.0, "yaw_deg": 10.0, "pitch_deg": 25.0}
    assert data["seed"] == 42
    assert len(data["panels"]) == 3

    total_cards = 0
    for panel in data["panels"]:
        assert set(panel) == {"panel_id", "orientation_deg", "quad_px", "cards"}
        assert panel["orientation_deg"] in (0, 45, 90)
        assert len(panel["quad_px"]) == 4 and all(len(p) == 2 for p in panel["quad_px"])
        for card in panel["cards"]:
            assert set(card) == {"x", "y", "color", "cell_center_px"}
            assert 1 <= card["x"] <= 10 and 1 <= card["y"] <= 10
            assert len(card["cell_center_px"]) == 2
        total_cards += len(panel["cards"])
    assert total_cards == 10


def test_write_scene_json_keeps_polish_color_names(tmp_path):
    specs = sample_scene(panel_count=1, seed=3)
    cam = Camera(4000.0, 0.0, 25.0)
    out = tmp_path / "s.json"
    write_scene_json(out, "s.png", (1920, 1080), cam, 3, specs)
    raw = out.read_text(encoding="utf-8")
    assert "\\u" not in raw  # ensure_ascii=False -> real ł/ż/ó, not escapes
