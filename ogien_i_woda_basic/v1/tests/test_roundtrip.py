from dataclasses import replace

import pytest

from generator.groundtruth import build_scene_data, Camera
from generator.roundtrip import generate_and_detect, print_stats
from generator.sampler import Card, PanelSpec, sample_scene
from generator.scene import compose_scene

import main


def _detect_xy(spec: PanelSpec, pitch: float = 15.0) -> set[tuple[int, int, str]]:
    img, _ = compose_scene([spec], width=1920, height=1080, distance=4000, yaw=0.0, pitch=pitch)
    out: set[tuple[int, int, str]] = set()
    for panel in main.detect_panels(img):
        for a in main.detect_anomalies_on_panel(img, panel):
            out.add((a.grid_x, a.grid_y, a.color))
    return out


def test_roundtrip_harness_runs_and_segments_three_panels():
    stats = generate_and_detect(panel_count=3, seed=42, yaw=0.0, pitch=20.0)
    print_stats(stats)
    assert stats["ground_truth_count"] == 10
    assert stats["panels_detected"] == 3
    assert 0.0 <= stats["precision"] <= 1.0
    assert 0.0 <= stats["recall"] <= 1.0


@pytest.mark.parametrize("orientation", [0, 45, 90])
def test_white_marker_fixes_every_orientation(orientation):
    """White (1,1) marker + side-length corner ordering => correct X,Y at 0/45/90 deg."""
    cards = [Card(2, 3, "czerwona"), Card(7, 4, "niebieska"),
             Card(4, 8, "zielona"), Card(9, 6, "fioletowa")]
    spec = PanelSpec(panel_id=1, orientation_deg=orientation, cards=cards)
    gt = {(c.x, c.y, c.color) for c in cards}
    det = _detect_xy(spec)
    assert det == gt, f"orientation {orientation}: {det ^ gt}"


@pytest.mark.parametrize("seed", [1, 2, 3, 7, 10, 11, 20, 42, 99])
def test_roundtrip_precision_recall_across_seeds(seed):
    """Fixed-seed regression: 3 panels, mixed orientations, >= 0.9 precision AND recall."""
    stats = generate_and_detect(panel_count=3, seed=seed, yaw=0.0, pitch=20.0)
    print_stats(stats)
    assert stats["panels_detected"] == 3
    assert stats["recall"] >= 0.9, f"seed {seed}: recall {stats['recall']:.0%}"
    assert stats["precision"] >= 0.9, f"seed {seed}: precision {stats['precision']:.0%}"


def test_per_colour_recall_orientation0():
    """Every one of the 6 colours round-trips; overall recall is high on flat panels."""
    seen: dict[str, list[int]] = {}
    for seed in (7, 11, 42):
        specs = [replace(s, orientation_deg=0) for s in sample_scene(3, seed)]
        img, _ = compose_scene(specs, 1920, 1080, distance=4000, yaw=0.0, pitch=20.0)
        gt = build_scene_data("x", (1920, 1080), Camera(4000, 0.0, 20.0), seed, specs)
        gtset = {(p.panel_id, c.x, c.y, c.color) for p in gt.panels for c in p.cards}
        det = set()
        for panel in main.detect_panels(img):
            for a in main.detect_anomalies_on_panel(img, panel):
                det.add((a.panel_id, a.grid_x, a.grid_y, a.color))
        for (_, _, _, colour) in gtset:
            seen.setdefault(colour, [0, 0])
        for tup in gtset:
            seen[tup[3]][1] += 1
            if tup in det:
                seen[tup[3]][0] += 1

    for colour, (hit, total) in seen.items():
        assert hit / total >= 0.5, f"{colour}: {hit}/{total}"
