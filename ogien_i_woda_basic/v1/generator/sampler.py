from __future__ import annotations

import random
from dataclasses import dataclass

from generator.regulation import COLORS, CORNER_MARKER_CELL


@dataclass(frozen=True)
class Card:
    x: int
    y: int
    color: str


@dataclass(frozen=True)
class PanelSpec:
    panel_id: int
    orientation_deg: int
    cards: list[Card]


_COLOR_NAMES = [name for name, _ in COLORS]
# Exclude corner marker cell (1,1) so no card is sampled there
_GRID_CELLS = [(x, y) for x in range(1, 11) for y in range(1, 11) if (x, y) != CORNER_MARKER_CELL]


def _panel_card_counts(panel_count: int, rng: random.Random) -> list[int]:
    if panel_count == 3:
        counts = [3, 3, 4]
        rng.shuffle(counts)
        return counts
    if panel_count in (1, 2):
        return [rng.choice((3, 4)) for _ in range(panel_count)]
    raise ValueError("panel_count must be 1, 2, or 3")


def _sample_panel(panel_id: int, card_count: int, rng: random.Random) -> PanelSpec:
    cells = rng.sample(_GRID_CELLS, k=card_count)
    colors = rng.sample(_COLOR_NAMES, k=card_count)
    cards = [Card(x=x, y=y, color=color) for (x, y), color in zip(cells, colors)]
    return PanelSpec(panel_id=panel_id, orientation_deg=rng.choice((0, 45, 90)), cards=cards)


def sample_scene(panel_count: int = 3, seed: int | None = None) -> list[PanelSpec]:
    """Sample a valid synthetic scene according to the panel-generator spec."""
    rng = random.Random(seed)
    counts = _panel_card_counts(panel_count, rng)
    return [_sample_panel(panel_id=i + 1, card_count=count, rng=rng) for i, count in enumerate(counts)]
