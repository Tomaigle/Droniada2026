from generator.roundtrip import generate_and_detect, print_stats


def test_roundtrip_harness_runs_and_segments_three_panels():
    """The generate -> detect harness works and the scene is segmentable into 3 panels."""
    stats = generate_and_detect(panel_count=3, seed=42, yaw=0.0, pitch=20.0)
    print_stats(stats)

    assert stats["ground_truth_count"] == 10
    assert stats["panels_detected"] == 3, "scene must render 3 separated, contrasting panels"
    assert 0.0 <= stats["precision"] <= 1.0
    assert 0.0 <= stats["recall"] <= 1.0


def test_roundtrip_localisation_regression():
    """After the main.py grid-axis fix, the detector localises the majority of cards.

    seed=20 is a fixed regression point: orientations [0, 90, 0] (so a rotated panel is
    exercised, not a best-case all-0 scene). Cross-seed robustness (rotated panels,
    marginal HSV colours) is still weak -- tracked as T9.
    """
    stats = generate_and_detect(panel_count=3, seed=20, yaw=0.0, pitch=20.0)
    print_stats(stats)

    assert stats["recall"] >= 0.6, f"Recall {stats['recall']:.1%} regressed below 60%"
    assert stats["precision"] >= 0.6, f"Precision {stats['precision']:.1%} regressed below 60%"
