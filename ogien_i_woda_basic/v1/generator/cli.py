from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from generator.groundtruth import Camera, write_scene_json
from generator.sampler import sample_scene
from generator.scene import compose_scene


def main() -> None:
    """CLI entrypoint: generate a scene and write image + JSON ground truth."""
    parser = argparse.ArgumentParser(description="Generate synthetic fire-detection training scenes.")
    parser.add_argument(
        "--panels",
        type=int,
        choices=[1, 2, 3],
        default=3,
        help="Number of panels (default 3).",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility.")
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output path stem (no extension); .png and .json appended.",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=4000.0,
        help="Camera distance in mm (default 4000).",
    )
    parser.add_argument("--yaw", type=float, default=0.0, help="Camera yaw in degrees (default 0).")
    parser.add_argument(
        "--pitch",
        type=float,
        default=25.0,
        help="Camera pitch in degrees (default 25).",
    )
    parser.add_argument("--grid", action="store_true", help="Draw grid on panels (debug).")

    args = parser.parse_args()

    # Generate scene
    specs = sample_scene(panel_count=args.panels, seed=args.seed)
    img, _ = compose_scene(
        specs, width=1920, height=1080, distance=args.distance, yaw=args.yaw, pitch=args.pitch, show_grid=args.grid
    )

    # Write image
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = out_path.with_suffix(".png")
    cv2.imwrite(str(png_path), img)

    # Write JSON ground truth
    json_path = out_path.with_suffix(".json")
    camera = Camera(distance_mm=args.distance, yaw_deg=args.yaw, pitch_deg=args.pitch)
    write_scene_json(json_path, png_path.name, (1920, 1080), camera, args.seed, specs)

    print(f"Image: {png_path}")
    print(f"JSON:  {json_path}")


if __name__ == "__main__":
    main()
