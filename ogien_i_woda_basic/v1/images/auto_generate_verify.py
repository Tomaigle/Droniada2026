import argparse
import json
from pathlib import Path
import sys

import cv2

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import main
from images import generate_panel_images as gen


def save_text_report(path: Path, report: dict) -> None:
    lines = []
    lines.append(f"Generated image: {report['generated_image']}")
    lines.append(f"Overlay image: {report['overlay_image']}")
    lines.append("\nParameters:")
    for key, value in report['parameters'].items():
        lines.append(f"  {key}: {value}")
    lines.append("\nDetection summary:")
    lines.append(f"  panels_detected: {report['detection']['panels_detected']}")
    lines.append(f"  total_anomalies: {report['detection']['total_anomalies']}")
    for panel in report['detection']['panels']:
        lines.append(f"  Panel {panel['panel_id']}: {panel['anomalies_count']} anomalies")
        for anomaly in panel['anomalies']:
            lines.append(f"    - {anomaly['color']} at ({anomaly['grid_x']},{anomaly['grid_y']}) confidence={anomaly['confidence']}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Generuje panel, uruchamia detekcję i zapisuje raport." )
    parser.add_argument("--output-name", default="auto_generated.png",
                        help="Nazwa wygenerowanego pliku obrazu")
    parser.add_argument("--output-dir", default=str(gen.DEFAULT_OUTPUT_DIR),
                        help="Katalog zapisu dla obrazu, overlay i raportu")
    parser.add_argument("--width", type=int, default=1920,
                        help="Szerokość obrazu wyjściowego")
    parser.add_argument("--height", type=int, default=1080,
                        help="Wysokość obrazu wyjściowego")
    parser.add_argument("--distance", type=float, default=4000.0,
                        help="Dystans kamery do paneli w mm")
    parser.add_argument("--drone-yaw", type=float, default=10.0,
                        help="Kąt yaw kamery w stopniach")
    parser.add_argument("--drone-pitch", type=float, default=25.0,
                        help="Kąt pitch kamery w stopniach")
    parser.add_argument("--panel-rotation", type=float, default=5.0,
                        help="Obrót panelu w płaszczyźnie w stopniach")
    parser.add_argument("--panel-tilt", type=float, default=10.0,
                        help="Kąt nachylenia panelu w stopniach")
    parser.add_argument("--show-grid", action="store_true",
                        help="Rysuj siatkę na wygenerowanym panelu")
    parser.add_argument("--random", action="store_true", default=True,
                        help="Włącz losowe rozmieszczenie anomalii")
    parser.add_argument("--seed", type=int, default=42,
                        help="Ziarno generatora losowego")
    parser.add_argument("--panel-count", type=int, choices=[1, 2, 3], default=1,
                        help="Ilość paneli na obrazie")
    parser.add_argument("--min-anomalies-per-panel", type=int, default=3,
                        help="Minimalna liczba anomalii na panel")
    parser.add_argument("--max-anomalies-per-panel", type=int, default=4,
                        help="Maksymalna liczba anomalii na panel")
    parser.add_argument("--anomaly", action="append", type=gen.parse_anomaly,
                        help="Ręczna specyfikacja anomalii: panel_id,x,y,color_hex")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_image_path = output_dir / args.output_name
    overlay_image_path = output_dir / f"overlay_{args.output_name}"
    report_json_path = output_dir / f"report_{Path(args.output_name).stem}.json"
    report_txt_path = output_dir / f"report_{Path(args.output_name).stem}.txt"

    anomaly_specs = args.anomaly or []
    panels = gen.generate_panels(anomaly_specs, args.random, args.seed, gen.DEFAULT_COLORS,
                                 args.min_anomalies_per_panel, args.max_anomalies_per_panel,
                                 args.panel_count)

    x_positions = [0.25, 0.50, 0.75][:args.panel_count]
    panel_positions = [(args.width * x, args.height * 0.65) for x in x_positions]
    panel_specs = []
    for idx in range(args.panel_count):
        panel_specs.append({
            'panel_id': idx + 1,
            'center': panel_positions[idx],
            'rotation': args.panel_rotation,
            'tilt': args.panel_tilt,
            'camera_yaw': args.drone_yaw,
            'camera_pitch': args.drone_pitch,
            'anomalies': panels[idx + 1],
        })

    generated = gen.compose_image((args.width, args.height), panel_specs,
                                  panel_width_px=gen.PANEL_PIXELS[0],
                                  panel_height_px=gen.PANEL_PIXELS[1],
                                  show_grid=args.show_grid,
                                  distance=args.distance)
    cv2.imwrite(str(output_image_path), generated)

    image = cv2.imread(str(output_image_path))
    detected_panels = main.detect_panels(image)
    all_anomalies = []
    for panel in detected_panels:
        found = main.detect_anomalies_on_panel(image, panel)
        panel.anomalies = found
        all_anomalies.extend(found)

    overlay = main.draw_debug(image, detected_panels, all_anomalies)
    cv2.imwrite(str(overlay_image_path), overlay)

    report = {
        'generated_image': str(output_image_path),
        'overlay_image': str(overlay_image_path),
        'parameters': {
            'width': args.width,
            'height': args.height,
            'distance': args.distance,
            'drone_yaw': args.drone_yaw,
            'drone_pitch': args.drone_pitch,
            'panel_rotation': args.panel_rotation,
            'panel_tilt': args.panel_tilt,
            'panel_count': args.panel_count,
            'min_anomalies_per_panel': args.min_anomalies_per_panel,
            'max_anomalies_per_panel': args.max_anomalies_per_panel,
            'seed': args.seed,
            'show_grid': args.show_grid,
            'anomaly_specs': anomaly_specs,
        },
        'detection': {
            'panels_detected': len(detected_panels),
            'total_anomalies': len(all_anomalies),
            'panels': [
                {
                    'panel_id': panel.id,
                    'anomalies_count': len(panel.anomalies),
                    'anomalies': [
                        {
                            'color': a.color,
                            'grid_x': a.grid_x,
                            'grid_y': a.grid_y,
                            'confidence': a.confidence,
                            'pixel_center': a.pixel_center,
                        }
                        for a in panel.anomalies
                    ],
                }
                for panel in detected_panels
            ],
        },
    }

    report_json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    save_text_report(report_txt_path, report)

    print(f"Wygenerowano: {output_image_path}")
    print(f"Zapisano overlay: {overlay_image_path}")
    print(f"Zapisano raporty: {report_json_path}, {report_txt_path}")


if __name__ == "__main__":
    main_cli()
