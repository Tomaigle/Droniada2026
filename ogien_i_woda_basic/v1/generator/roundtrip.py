from __future__ import annotations

import sys
from pathlib import Path

import cv2

# Add v1 to sys.path so we can import main and groundtruth
_v1_path = Path(__file__).parent.parent
if str(_v1_path) not in sys.path:
    sys.path.insert(0, str(_v1_path))

import main
from generator.groundtruth import Camera, build_scene_data
from generator.sampler import sample_scene
from generator.scene import compose_scene


def generate_and_detect(
    panel_count: int = 3,
    seed: int | None = 42,
    distance: float = 4000.0,
    yaw: float = 0.0,
    pitch: float = 25.0,
) -> dict:
    """Generate a scene and run detections; return stats including precision/recall."""
    # Generate scene
    specs = sample_scene(panel_count=panel_count, seed=seed)
    img, _ = compose_scene(specs, width=1920, height=1080, distance=distance, yaw=yaw, pitch=pitch)

    # Build ground truth structure (no JSON write)
    camera = Camera(distance_mm=distance, yaw_deg=yaw, pitch_deg=pitch)
    gt = build_scene_data("synthetic.png", (1920, 1080), camera, seed, specs)

    # Run detection
    panels = main.detect_panels(img)
    panels_detected = len(panels)
    detections = []
    for panel in panels:
        anomalies = main.detect_anomalies_on_panel(img, panel)
        for anom in anomalies:
            detections.append((anom.panel_id, anom.grid_x, anom.grid_y, anom.color))

    # Build ground truth set (panel_id, x, y, color)
    ground_truth = set()
    for panel_data in gt.panels:
        for card in panel_data.cards:
            ground_truth.add((panel_data.panel_id, card.x, card.y, card.color))

    # Compute metrics
    detections_set = set(detections)
    tp = len(detections_set & ground_truth)
    fp = len(detections_set - ground_truth)
    fn = len(ground_truth - detections_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Per-color accuracy
    per_color_hits = {}
    for color in main.COLOR_RANGES:
        gt_color = {(p, x, y, c) for (p, x, y, c) in ground_truth if c == color}
        det_color = {(p, x, y, c) for (p, x, y, c) in detections_set if c == color}
        tp_color = len(gt_color & det_color)
        fn_color = len(gt_color - det_color)
        recall_color = tp_color / (tp_color + fn_color) if (tp_color + fn_color) > 0 else 0.0
        per_color_hits[color] = {"detected": tp_color, "missed": fn_color, "recall": recall_color}

    return {
        "ground_truth_count": len(ground_truth),
        "panels_detected": panels_detected,
        "detections_count": len(detections_set),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "per_color": per_color_hits,
    }


def print_stats(stats: dict) -> None:
    """Print roundtrip statistics."""
    print(f"\n=== Roundtrip Results ===")
    print(f"Ground Truth:  {stats['ground_truth_count']} cards")
    print(f"Panels found:  {stats['panels_detected']}")
    print(f"Detections:    {stats['detections_count']} cards")
    print(f"TP: {stats['tp']}, FP: {stats['fp']}, FN: {stats['fn']}")
    print(f"Precision: {stats['precision']:.1%}, Recall: {stats['recall']:.1%}")
    print(f"\nPer-color recall:")
    for color, hits in stats["per_color"].items():
        print(f"  {color:>12s}: {hits['detected']}/{hits['detected'] + hits['missed']} "
              f"({hits['recall']:.1%})")
