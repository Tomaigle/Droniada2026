from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from generator.sampler import PanelSpec
from generator.scene import panel_corner_quad_px, panel_homography, project_cell_center_px


@dataclass
class Camera:
    distance_mm: float
    yaw_deg: float
    pitch_deg: float


@dataclass
class CardData:
    x: int
    y: int
    color: str
    cell_center_px: list[float]


@dataclass
class PanelData:
    panel_id: int
    orientation_deg: int
    quad_px: list[list[float]]
    cards: list[CardData]


@dataclass
class SceneData:
    image: str
    image_size: list[int]
    camera: Camera
    seed: int | None
    panels: list[PanelData]


def build_scene_data(
    image_name: str,
    image_size: tuple[int, int],
    camera: Camera,
    seed: int | None,
    specs: list[PanelSpec],
) -> SceneData:
    """Build the ground-truth structure for a scene. Homographies are rebuilt from `camera`
    with the SAME formula `compose_scene` uses, so pixel coords line up with the image."""
    width, height = image_size
    panels: list[PanelData] = []
    for idx, spec in enumerate(sorted(specs, key=lambda s: s.panel_id)):
        h = panel_homography(
            spec, idx, len(specs), width, height,
            camera.distance_mm, camera.yaw_deg, camera.pitch_deg,
        )
        if h is None:
            continue
        cards = [
            CardData(card.x, card.y, card.color, project_cell_center_px(h, card.x, card.y))
            for card in spec.cards
        ]
        panels.append(PanelData(spec.panel_id, spec.orientation_deg, panel_corner_quad_px(h), cards))

    return SceneData(
        image=image_name,
        image_size=[width, height],
        camera=camera,
        seed=seed,
        panels=panels,
    )


def write_scene_json(
    path: str | Path,
    image_name: str,
    image_size: tuple[int, int],
    camera: Camera,
    seed: int | None,
    specs: list[PanelSpec],
) -> None:
    """Write ground-truth JSON for a generated scene, matching the spec schema exactly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scene = build_scene_data(image_name, image_size, camera, seed, specs)
    path.write_text(json.dumps(asdict(scene), indent=2, ensure_ascii=False), encoding="utf-8")
