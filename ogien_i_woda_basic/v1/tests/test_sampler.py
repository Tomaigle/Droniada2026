from generator.sampler import sample_scene


def test_sample_scene_three_panels_total_cards_and_panel_counts():
    specs = sample_scene(panel_count=3, seed=42)
    assert len(specs) == 3
    counts = [len(spec.cards) for spec in specs]
    assert sorted(counts) == [3, 3, 4]
    assert sum(counts) == 10
    assert {spec.orientation_deg for spec in specs}.issubset({0, 45, 90})

    for spec in specs:
        assert len({card.color for card in spec.cards}) == len(spec.cards)
        assert len({(card.x, card.y) for card in spec.cards}) == len(spec.cards)
        assert set(spec.cards[0].__dict__.keys()) == {"x", "y", "color"}


def test_sample_scene_seed_is_deterministic():
    a = sample_scene(panel_count=3, seed=123)
    b = sample_scene(panel_count=3, seed=123)
    assert a == b


def test_sample_scene_debug_panel_counts_are_valid():
    for panel_count in (1, 2):
        specs = sample_scene(panel_count=panel_count, seed=7)
        assert len(specs) == panel_count
        for spec in specs:
            assert 3 <= len(spec.cards) <= 4
            assert len({card.color for card in spec.cards}) == len(spec.cards)
            assert len({(card.x, card.y) for card in spec.cards}) == len(spec.cards)


def test_sample_scene_rejects_invalid_panel_count():
    try:
        sample_scene(panel_count=4, seed=0)
        raise AssertionError("panel_count=4 should raise ValueError")
    except ValueError:
        pass
