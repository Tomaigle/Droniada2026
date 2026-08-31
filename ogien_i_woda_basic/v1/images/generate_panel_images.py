import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np

DEFAULT_COLORS = [
    "#FF0000", "#C00000", "#FFA500", "#FF8C00", "#FFFF00",
    "#FFD700", "#008000", "#00A000", "#0000FF", "#0040FF"
]

PANEL_MM = (2000, 1000)
CELL_MM = (200, 100)
GRID_SIZE = (10, 10)
PANEL_PIXELS = (1000, 500)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "generated_images"
MIN_ANOMALIES_PER_PANEL = 3
MAX_ANOMALIES_PER_PANEL = 4
MAX_UNIQUE_COLORS = len(DEFAULT_COLORS)
DEFAULT_PANEL_POSITIONS = [(0.25, 0.65), (0.50, 0.65), (0.75, 0.65)]
DEFAULT_PANEL_ROTATIONS = [0.0, 0.0, 0.0]
DEFAULT_PANEL_TILTS = [0.0, 0.0, 0.0]


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.strip().lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(c * 2 for c in hex_color)
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)


def rotate_quad(quad: np.ndarray, center: tuple[float, float], angle_deg: float) -> np.ndarray:
    if angle_deg == 0.0:
        return quad
    angle = math.radians(angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    cx, cy = center
    rotated = []
    for x, y in quad:
        dx = x - cx
        dy = y - cy
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        rotated.append([cx + rx, cy + ry])
    return np.array(rotated, dtype=np.float32)


def build_panel_projection(panel_center: tuple[float, float],
                            panel_rotation_deg: float,
                            panel_tilt_deg: float,
                            panel_width_px: int,
                            panel_height_px: int,
                            width: int,
                            height: int,
                            distance: float) -> np.ndarray:
    cx, cy = panel_center
    scale = max(0.3, min(1.2, 4000.0 / max(1.0, distance)))
    w = panel_width_px * scale
    h = panel_height_px * scale

    skew = math.tan(math.radians(panel_rotation_deg)) * w * 0.15
    top_scale = max(0.25, 1.0 - abs(panel_tilt_deg) / 120.0)
    top_h = h * top_scale

    left_x = cx - w * 0.5
    right_x = cx + w * 0.5
    bottom_y = cy + h * 0.5
    top_y = cy - top_h * 0.5

    quad = np.array([
        [left_x, bottom_y],
        [right_x, bottom_y],
        [right_x + skew, top_y],
        [left_x + skew, top_y],
    ], dtype=np.float32)

    return rotate_quad(quad, (cx, cy), panel_rotation_deg)


def validate_panel_dimensions(panel_width_px: int,
                                panel_height_px: int) -> None:
    if GRID_SIZE != (10, 10):
        raise ValueError("Panel musi być zawsze siatką 10x10.")
    if panel_width_px * 2 != panel_height_px * 4:
        raise ValueError("Panel musi mieć proporcje 2:1 (2m x 1m).")
    if panel_width_px % GRID_SIZE[0] != 0 or panel_height_px % GRID_SIZE[1] != 0:
        raise ValueError("Panel musi być podzielny na jednolite pola siatki 10x10.")


def build_panel_image(panel_width_px: int,
                      panel_height_px: int,
                      anomalies: dict[tuple[int, int], tuple[int, int, int]],
                      show_grid: bool) -> np.ndarray:
    validate_panel_dimensions(panel_width_px, panel_height_px)
    img = np.zeros((panel_height_px, panel_width_px, 3), dtype=np.uint8)
    cell_w = panel_width_px // GRID_SIZE[0]
    cell_h = panel_height_px // GRID_SIZE[1]

    for (gx, gy), color in anomalies.items():
        x0 = (gx - 1) * cell_w
        y0 = (gy - 1) * cell_h
        x1 = x0 + cell_w
        y1 = y0 + cell_h
        cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness=-1)

    if show_grid:
        line_color = (60, 60, 60)
        for ix in range(1, GRID_SIZE[0]):
            x = ix * cell_w
            cv2.line(img, (x, 0), (x, panel_height_px), line_color, 1)
        for iy in range(1, GRID_SIZE[1]):
            y = iy * cell_h
            cv2.line(img, (0, y), (panel_width_px, y), line_color, 1)

    return img


def generate_panels(anomaly_specs: list[tuple[int, int, int, str]],
                    random_mode: bool,
                    seed: int | None,
                    colors: list[str],
                    min_anomalies: int,
                    max_anomalies: int,
                    panel_count: int) -> dict[int, dict[tuple[int, int], tuple[int, int, int]]]:
    if seed is not None:
        random.seed(seed)

    panels: dict[int, dict[tuple[int, int], tuple[int, int, int]]] = {
        1: {},
        2: {},
        3: {},
    }

    if anomaly_specs:
        used_colors: set[str] = set()
        for panel_id, gx, gy, color_hex in anomaly_specs:
            if panel_id not in panels:
                raise ValueError(f"Nieprawidłowy panel_id: {panel_id}")
            if not (1 <= gx <= GRID_SIZE[0] and 1 <= gy <= GRID_SIZE[1]):
                raise ValueError(f"Nieprawidłowe współrzędne: {gx},{gy}")
            if color_hex in used_colors:
                raise ValueError(f"Kolor {color_hex} powtarza się w specyfikacji anomalii")
            used_colors.add(color_hex)
            panels[panel_id][(gx, gy)] = hex_to_bgr(color_hex)
    else:
        used_colors = set()

    if random_mode:
        existing = set((panel_id, gx, gy) for panel_id in panels for (gx, gy) in panels[panel_id])
        panel_order = [pid for pid in range(1, panel_count + 1)]
        requested = [random.randint(min_anomalies, max_anomalies) for _ in panel_order]
        total_requested = sum(requested)
        total_allowed = min(MAX_UNIQUE_COLORS - len(used_colors), sum(requested))

        if total_allowed <= 0:
            return panels

        # Reduce the requested count if there are not enough unique colors
        while sum(requested) > total_allowed:
            idx = random.randrange(len(requested))
            if requested[idx] > min_anomalies:
                requested[idx] -= 1

        available_colors = [c for c in colors if c not in used_colors]
        random.shuffle(available_colors)

        all_positions = [
            (panel_id, gx, gy)
            for panel_id in panel_order
            for gx in range(1, GRID_SIZE[0] + 1)
            for gy in range(1, GRID_SIZE[1] + 1)
        ]
        random.shuffle(all_positions)

        for panel_id, count in zip(panel_order, requested):
            added = 0
            for (pid, gx, gy) in all_positions:
                if pid != panel_id or (pid, gx, gy) in existing:
                    continue
                if added >= count:
                    break
                if not available_colors:
                    break
                color_hex = available_colors.pop()
                panels[pid][(gx, gy)] = hex_to_bgr(color_hex)
                existing.add((pid, gx, gy))
                added += 1

    return panels


def build_panel_projection(panel_center: tuple[float, float],
                            camera_yaw_deg: float,
                            camera_pitch_deg: float,
                            panel_rotation_deg: float,
                            panel_tilt_deg: float,
                            panel_width_px: int,
                            panel_height_px: int,
                            width: int,
                            height: int,
                            distance: float) -> np.ndarray:
    cx, cy = panel_center
    scale = max(0.3, min(1.2, 4000.0 / max(1.0, distance)))
    w = panel_width_px * scale
    h = panel_height_px * scale

    yaw_skew = math.tan(math.radians(camera_yaw_deg)) * w * 0.15
    pitch_scale = max(0.25, 1.0 - abs(camera_pitch_deg) / 90.0 * 0.6)
    tilt_scale = max(0.4, 1.0 - abs(panel_tilt_deg) / 90.0 * 0.5)
    top_h = h * pitch_scale * tilt_scale

    left_x = cx - w * 0.5
    right_x = cx + w * 0.5
    bottom_y = cy + h * 0.5
    top_y = cy - top_h * 0.5

    quad = np.array([
        [left_x, bottom_y],
        [right_x, bottom_y],
        [right_x + yaw_skew, top_y],
        [left_x + yaw_skew, top_y],
    ], dtype=np.float32)

    return rotate_quad(quad, (cx, cy), panel_rotation_deg)


def compose_image(output_size: tuple[int, int],
                  panel_specs: list[dict],
                  panel_width_px: int,
                  panel_height_px: int,
                  show_grid: bool,
                  distance: float) -> np.ndarray:
    width, height = output_size
    output = np.zeros((height, width, 3), dtype=np.uint8)

    panels_sorted = sorted(panel_specs, key=lambda p: p['center'][1], reverse=True)
    for spec in panels_sorted:
        projection = build_panel_projection(
            spec['center'],
            spec['camera_yaw'],
            spec['camera_pitch'],
            spec['rotation'],
            spec['tilt'],
            panel_width_px,
            panel_height_px,
            width,
            height,
            distance,
        )

        if np.any(np.isnan(projection)):
            continue

        src = np.array([[0.0, 0.0], [panel_width_px, 0.0], [panel_width_px, panel_height_px], [0.0, panel_height_px]], dtype=np.float32)
        dst = projection
        H = cv2.getPerspectiveTransform(src, dst)

        panel_img = build_panel_image(panel_width_px, panel_height_px, spec['anomalies'], show_grid)
        warped = cv2.warpPerspective(panel_img, H, (width, height), flags=cv2.INTER_LINEAR)

        mask = cv2.warpPerspective(np.ones((panel_height_px, panel_width_px), dtype=np.uint8) * 255,
                                   H, (width, height), flags=cv2.INTER_NEAREST)
        mask_bool = mask.astype(bool)
        output[mask_bool] = warped[mask_bool]

    return output


def parse_anomaly(arg: str) -> tuple[int, int, int, str]:
    try:
        panel_id, x, y, color = arg.split(",")
        return int(panel_id.strip()), int(x.strip()), int(y.strip()), color.strip()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Anomalia musi mieć format panel_id,x,y,color_hex") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generuje obrazy paneli z anomaliami i perspektywicznym rzutowaniem.")
    parser.add_argument("--output", default="generated.png",
                        help="Nazwa pliku wyjściowego PNG")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help="Katalog wyjściowy, domyślnie generated_images")
    parser.add_argument("--width", type=int, default=1920,
                        help="Szerokość obrazu wyjściowego")
    parser.add_argument("--height", type=int, default=1080,
                        help="Wysokość obrazu wyjściowego")
    parser.add_argument("--distance", type=float, default=4000.0,
                        help="Dystans kamery do paneli w mm")
    parser.add_argument("--drone-yaw", type=float, default=0.0,
                        help="Kąt yaw (boczny) kamery w stopniach")
    parser.add_argument("--drone-pitch", type=float, default=30.0,
                        help="Kąt pitch (pionowy) kamery w stopniach")
    parser.add_argument("--panel-rotation", type=float, default=0.0,
                        help="Obrót panelu w płaszczyźnie względem osi Z w stopniach")
    parser.add_argument("--panel-tilt", type=float, default=0.0,
                        help="Kąt nachylenia panelu względem podłoża w stopniach")
    parser.add_argument("--show-grid", action="store_true",
                        help="Rysuj siatkę na panelach")
    parser.add_argument("--random", action="store_true",
                        help="Wygeneruj losowe anomalie na panelach")
    parser.add_argument("--seed", type=int, default=None,
                        help="Ziarno generatora losowego")
    parser.add_argument("--panel-count", type=int, choices=[1, 2, 3], default=1,
                        help="Liczba paneli do wygenerowania")
    parser.add_argument("--min-anomalies-per-panel", type=int, default=3,
                        help="Minimalna liczba anomalii na panel")
    parser.add_argument("--max-anomalies-per-panel", type=int, default=4,
                        help="Maksymalna liczba anomalii na panel")
    parser.add_argument("--anomaly", action="append", type=parse_anomaly,
                        help="Specyfikacja anomalii: panel_id,x,y,color_hex")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output

    x_positions = [0.25, 0.50, 0.75][:args.panel_count]
    panel_positions = [(args.width * x, args.height * 0.65) for x in x_positions]
    panel_rotations = [args.panel_rotation] * args.panel_count
    panel_tilts = [args.panel_tilt] * args.panel_count

    if args.min_anomalies_per_panel < 0 or args.max_anomalies_per_panel < args.min_anomalies_per_panel:
        raise ValueError("Nieprawidłowe wartości min/max anomalii na panel.")

    anomaly_specs = args.anomaly or []
    panels = generate_panels(anomaly_specs, args.random, args.seed, DEFAULT_COLORS,
                             args.min_anomalies_per_panel,
                             args.max_anomalies_per_panel,
                             args.panel_count)

    panel_specs = []
    for idx in range(args.panel_count):
        panel_id = idx + 1
        panel_specs.append({
            'panel_id': panel_id,
            'center': panel_positions[idx],
            'rotation': panel_rotations[idx],
            'tilt': panel_tilts[idx],
            'camera_yaw': args.drone_yaw,
            'camera_pitch': args.drone_pitch,
            'anomalies': panels[panel_id],
        })

    output = compose_image((args.width, args.height), panel_specs,
                           panel_width_px=1000, panel_height_px=500,
                           show_grid=args.show_grid,
                           distance=args.distance)
    cv2.imwrite(str(output_path), output)
    print(f"Zapisano obraz: {output_path}")


if __name__ == "__main__":
    main()
