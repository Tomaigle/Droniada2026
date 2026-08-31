---
description: "Spec for the Ogień i Woda BASIC synthetic panel generator. Load when editing anything under ogien_i_woda_basic/."
applyTo: "ogien_i_woda_basic/**"
---

# Ogień i Woda BASIC — Panel Generator Spec (authoritative)

Source of truth: `dokumentacja/Regulamin_konkursu_Droniada_Challenge_2026.pdf`, section
"ETAP BASIC - OGIEŃ" (pdftotext -layout lines ~623–690). This file restates it so you
do not have to re-read the PDF. If this file and your memory disagree, this file wins.

## What we are building

A synthetic image generator for the BASIC fire task: renders aerial-style images of PV
"panels" (black banners) with colored-card anomalies, plus a machine-readable
ground-truth JSON. Used to test/tune the detector in `ogien_i_woda_basic/v1/main.py`
and to train models later.

This is a TEST DATA GENERATOR, not the detector. Do not touch detection logic in
`main.py` except to import its constants/functions for the round-trip check.

## Regulation facts — NON-NEGOTIABLE

1. **3 black banners**, each **2 m wide × 1 m tall** (landscape).
2. Each banner is a **10 × 10 grid** of rectangles, each cell **20 cm (X) × 10 cm (Y)**.
3. **Coordinate system**: `(1,1)` = **bottom-left** corner. `X` = 1..10 left→right.
   `Y` = 1..10 **bottom→top**. So `(10,1)` = bottom-right, `(1,10)` = top-left,
   `(10,10)` = top-right, `(5,5)`/`(6,6)` = centre.
4. **Cell (x,y) origin**: `(x-1)·20 cm` from the LEFT edge, `(y-1)·10 cm` from the
   BOTTOM edge. Cell spans `20 cm` in X and `10 cm` in Y from there.
5. **Anomalies = colored cards.** Exactly **6 colors**, Polish names, must match
   `main.COLOR_RANGES` keys exactly:
   `czerwona, niebieska, fioletowa, zielona, żółta, pomarańczowa`.
6. Each card occupies **exactly one grid cell** (20×10 cm). Not multi-cell. Not sub-cell.
7. **Always 10 cards total** across the 3 panels, **3–4 per panel** (e.g. 3+3+4).
8. **Colors do not repeat within one panel.** They MAY repeat across panels.
9. Cards are placed at **random** cells (no two cards in the same cell on one panel).
10. Each panel may be independently oriented **0° / 45° / 90°** in-plane, and is mounted
    **≥ 1 m above ground** (→ perspective / tilt in the rendered view).
11. Detections are reported as `(X, Y)` + color per panel.

## Color table (use these; verify each round-trips through the detector)

| name (PL)     | hex      | note: must land inside main.COLOR_RANGES HSV band |
|---------------|----------|---------------------------------------------------|
| czerwona      | #FF0000  | H≈0                                               |
| pomarańczowa  | #FF7A00  | H≈18, inside 11–25                                |
| żółta         | #FFE000  | H≈30, inside 26–35                                |
| zielona       | #00A000  | H≈60, inside 36–85                                |
| niebieska     | #0000FF  | H≈120, inside 95–130                              |
| fioletowa     | #B000C0  | H≈150, inside 130–160, S high                     |

`main.py` uses OpenCV HSV (H 0–179). Colors are stored BGR in code. A unit test MUST
assert every generated card color is classified back to the same name by
`main.build_color_mask` — tune hex if not.

## Architecture — ONE package (BUILT: T1–T8 complete)

Everything lives in `ogien_i_woda_basic/v1/generator/`. Do NOT add a parallel script.
The old `images/*.py` generators were inconsistent with each other and this spec; they
were deleted in T8. History is in `COPILOT_BUILD_LOG.md`.

```
ogien_i_woda_basic/v1/generator/
  __init__.py
  regulation.py    # constants + addressing helpers + COLORS table. No cv2 needed.
  panel.py         # render_panel(cards, size_px, show_grid) -> flat top-down BGR image
  sampler.py       # sample_scene(panel_count, seed) -> list[PanelSpec] (rules 6-9)
  scene.py         # place panels (rot 0/45/90 + perspective), compose on background
  groundtruth.py   # dataclasses + JSON writer (schema below)
  cli.py           # argparse entrypoint -> writes <name>.png + <name>.json
  roundtrip.py     # generate -> main.detect_* -> precision/recall vs ground truth
ogien_i_woda_basic/v1/tests/
  test_regulation.py
  test_panel.py
  test_sampler.py
  test_roundtrip.py
```

### Geometry rules for `panel.py` (this is where past attempts broke)

Top-down raster image, OpenCV pixel origin = TOP-LEFT. For image `W×H` px,
`cell_w = W/10`, `cell_h = H/10`. Card at grid `(x, y)`:

```
x0 = (x - 1) * cell_w
y0 = (10 - y) * cell_h        # Y FLIP: grid y=1 is the BOTTOM row of the image
x1 = x0 + cell_w
y1 = y0 + cell_h
```

- Panel background is solid black `(0,0,0)`.
- **No outer border / frame** is drawn (the regulation banner has none; the detector
  keys on the black rectangle edge).
- `show_grid=True` draws thin interior grid lines only (debug); default `False`.
- Recommended `size_px=(1000, 500)` (keeps 2:1 and divisibility by 10).

### `sampler.py`

- `panel_count=3` (default) → total cards is exactly 10, per-panel count in `{3,4}`,
  partition chosen by RNG (valid partitions of 10 into 3 parts each 3–4: permutations
  of `[3,3,4]`).
- `panel_count` 1 or 2 is allowed for debugging: 3–4 cards per panel, total not forced.
- Per panel: sample distinct colors (≤6) and distinct cells. Seeded `random.Random(seed)`
  → identical output for identical seed.
- Return typed specs, not loose dicts.

### `scene.py`

- Perspective math (`build_panel_projection`, `rotate_quad`) was ported from the old
  `generate_panel_images.py` (camera_yaw/camera_pitch variant). Keeps `camera_yaw`,
  `camera_pitch`, `distance` knobs. Also: `BACKGROUND_BGR` grey ground and
  `PANEL_ON_SCREEN_FRAC` — panels must NOT be black-on-black or overlapping, or the
  detector segments the whole frame as one panel.
- Panel in-plane orientation supports exactly `0, 45, 90` degrees; that value goes
  into the ground truth.
- Output image default `1920×1080`.

### Ground-truth JSON schema (`groundtruth.py`)

```json
{
  "image": "scene_001.png",
  "image_size": [1920, 1080],
  "camera": { "distance_mm": 4000, "yaw_deg": 10.0, "pitch_deg": 25.0 },
  "seed": 42,
  "panels": [
    {
      "panel_id": 1,
      "orientation_deg": 45,
      "quad_px": [[x,y],[x,y],[x,y],[x,y]],
      "cards": [
        { "x": 3, "y": 7, "color": "czerwona", "cell_center_px": [cx, cy] }
      ]
    }
  ]
}
```

`quad_px` order: bottom-left, bottom-right, top-right, top-left (grid corners
`(1,1),(10,1),(10,10),(1,10)` projected to image pixels).

### `cli.py`

```
python -m generator.cli --panels 3 --seed 42 --out generated_images/scene_001 \
    --distance 4000 --yaw 10 --pitch 25 --grid
```

Writes `scene_001.png` + `scene_001.json` next to each other. `--panels` choices
`1,2,3` default `3`. Deterministic for a given `--seed`.

### `roundtrip.py`

Generate a scene → run `main.detect_panels` + `main.detect_anomalies_on_panel` on the
PNG → match detected `(panel, x, y, color)` against the JSON → print
precision / recall / per-color confusion. This is the acceptance signal for the whole
generator: colors and coordinates must survive a generate→detect cycle.

## Hard rules for the implementer

- Read a file fully before editing it (`read-before-edit.instructions.md`).
- No new dependencies beyond `opencv-python`, `numpy`, `pytest`.
- Type hints on every function; one-line docstring naming the regulation rule it
  enforces.
- Functions ≤ ~40 lines; if longer, split.
- Every task ends with its named test green: `python -m pytest ogien_i_woda_basic/v1/tests -q`.
- Do not delete the legacy scripts until the final task and until the new package
  passes all tests.
- Work strictly through the numbered tasks in
  `ogien_i_woda_basic/v1/COPILOT_BUILD_LOG.md`. Do one task per turn. Do not skip
  ahead. After finishing a task, stop and report what changed + test output.
