import os

os.environ["DISPLAY"] = ":0"
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import numpy as np
import time
import pyrealsense2 as rs

# Hailo Runtime API (hailort >= 4.17, dostarczany przez rpicam-apps / hailo-all)
from hailo_platform import (
    HEF,
    VDevice,
    HailoStreamInterface,
    InferVStreams,
    ConfigureParams,
    InputVStreamParams,
    OutputVStreamParams,
    FormatType,
)

# ── CONFIG ────────────────────────────────────────────────
# Model musi być skompilowany do formatu .hef przez Hailo Dataflow Compiler
# (DFC) lub pobrany z Hailo Model Zoo. Konwersja z .pt:
#   1. Eksport do ONNX:  yolo export model=new_best.pt format=onnx imgsz=640
#   2. Kompilacja HEF:   hailomz compile --ckpt new_best.onnx --hw-arch hailo8l
#      (lub hailo_dfc compile ... — zależy od wersji SDK)
MODEL_PATH   = "new_best.hef"   # ← plik .hef zamiast .pt
MODEL_W      = 640              # szerokość wejścia modelu [px]
MODEL_H      = 640              # wysokość wejścia modelu [px]

GRIP_DISTANCE  = 1.00   # [m] aktywacja chwytu gdy bliżej niż to
CENTER_THRESH  = 0.16   # [m] dopuszczalny błąd XY przed opadaniem
CONF_THRESHOLD = 0.05
KP             = 0.5    # wzmocnienie proporcjonalne dla komend prędkości
MAX_SPEED      = 0.5    # [m/s] maks. prędkość wysyłana do drona

# Rozmiar ramki z RealSense (kolor i głębia)
RS_W, RS_H = 640, 480
# ─────────────────────────────────────────────────────────


# ─── REALSENSE ────────────────────────────────────────────

def setup_realsense():
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, RS_W, RS_H, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, RS_W, RS_H, rs.format.z16,  30)
    profile = pipeline.start(cfg)
    align   = rs.align(rs.stream.color)

    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    intr = depth_profile.get_intrinsics()
    return pipeline, align, intr.fx, intr.fy


def get_depth(depth_frame, cx, cy, w, h):
    """Mediana głębokości z małego obszaru wokół punktu."""
    samples = []
    for dx in [-3, 0, 3]:
        for dy in [-3, 0, 3]:
            x = int(np.clip(cx + dx, 0, w - 1))
            y = int(np.clip(cy + dy, 0, h - 1))
            d = depth_frame.get_distance(x, y)
            if d > 0.05:
                samples.append(d)
    return float(np.median(samples)) if samples else 0.0


# ─── HAILO INFERENCE ──────────────────────────────────────

def setup_hailo(hef_path: str):
    """
    Inicjalizuje urządzenie Hailo, ładuje HEF i zwraca:
      network_group, input_vstreams_params, output_vstreams_params, infer_pipeline
    """
    hef    = HEF(hef_path)
    target = VDevice()                          # automatycznie wykrywa AI Hat 2

    configure_params = ConfigureParams.create_from_hef(
        hef, interface=HailoStreamInterface.PCIe
    )
    network_groups = target.configure(hef, configure_params)
    ng = network_groups[0]                      # zakładamy jeden network group

    in_params  = InputVStreamParams.make(ng,  format_type=FormatType.UINT8)
    out_params = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)

    return target, ng, in_params, out_params


def preprocess(frame_bgr: np.ndarray, model_w: int, model_h: int) -> np.ndarray:
    """
    Skaluje klatkę BGR do rozmiaru wejścia modelu.
    Hailo oczekuje UINT8 HWC (BGR lub RGB — zgodnie z HEF).
    """
    resized = cv2.resize(frame_bgr, (model_w, model_h))
    return resized  # shape: (model_h, model_w, 3), dtype uint8


def postprocess_yolov8(
    raw_outputs: dict,
    orig_w: int,
    orig_h: int,
    model_w: int,
    model_h: int,
    conf_thresh: float,
) -> list[tuple[int, int, int, int, float]]:
    """
    Parsuje wyjście sieci YOLOv8 z Hailo.

    Hailo Model Zoo kompiluje YOLOv8 z NMS wbudowanym w HEF.
    Wyjście ma nazwę 'yolov8_nms_postprocess' i format:
        [batch, num_detections, 7]  → [x1, y1, x2, y2, score, class_id, ?]
    lub płaski tensor z grupami po 5/6 floatów.

    Jeśli twój HEF ma inną architekturę wyjścia — dostosuj tę funkcję.
    Zwraca listę (x1, y1, x2, y2, conf) w pikselach oryginału.
    """
    detections = []

    for layer_name, data in raw_outputs.items():
        # data shape po squeeze: (N, 7) lub (N, 6)
        arr = np.squeeze(data)                  # usuń wymiar batch
        if arr.ndim == 1:
            arr = arr.reshape(-1, arr.shape[-1] if len(arr.shape) > 1 else 6)
        if arr.ndim != 2 or arr.shape[0] == 0:
            continue

        for row in arr:
            # Hailo NMS output: [y1, x1, y2, x2, score, class] (znormalizowane)
            # lub [x1, y1, x2, y2, score, class] — zależy od HEF
            # Poniżej zakładamy [y1, x1, y2, x2, score, class]:
            if len(row) < 5:
                continue
            y1n, x1n, y2n, x2n = row[0], row[1], row[2], row[3]
            score = float(row[4])

            if score < conf_thresh:
                continue

            # Denormalizacja do rozmiaru oryginału
            x1 = int(np.clip(x1n * orig_w, 0, orig_w - 1))
            y1 = int(np.clip(y1n * orig_h, 0, orig_h - 1))
            x2 = int(np.clip(x2n * orig_w, 0, orig_w - 1))
            y2 = int(np.clip(y2n * orig_h, 0, orig_h - 1))

            detections.append((x1, y1, x2, y2, score))

    return detections


# ─── KONTROLA ─────────────────────────────────────────────

def compute_mavlink_commands(err_x_m, err_y_m, depth):
    centered = abs(err_x_m) < CENTER_THRESH and abs(err_y_m) < CENTER_THRESH

    vx = float(np.clip(KP * err_x_m, -MAX_SPEED, MAX_SPEED))
    vy = float(np.clip(KP * err_y_m, -MAX_SPEED, MAX_SPEED))

    if centered and depth > GRIP_DISTANCE:
        vz = float(np.clip(KP * (depth - GRIP_DISTANCE), 0, MAX_SPEED))
    else:
        vz = 0.0

    grip = centered and depth < GRIP_DISTANCE
    return vx, vy, vz, grip


def print_mavlink(vx, vy, vz, grip, depth, err_x, err_y):
    ts = time.strftime("%H:%M:%S")
    if grip:
        print(f"[{ts}]  *** GRIP ***  depth={depth:.2f}m  err=({err_x:+.3f}, {err_y:+.3f})m")
    else:
        print(
            f"[{ts}]  MOVE  vx={vx:+.2f}  vy={vy:+.2f}  vz={vz:+.2f} m/s"
            f"  |  depth={depth:.2f}m  err=({err_x:+.3f}, {err_y:+.3f})m"
        )


# ─── OVERLAY ──────────────────────────────────────────────

def draw_overlay(frame, cx, cy, fw, fh, depth, vx, vy, vz, grip):
    fc  = (fw // 2, fh // 2)
    col = (0, 255, 0) if grip else (0, 200, 255)

    cv2.circle(frame, (cx, cy), 10, col, 2)
    cv2.line(frame, (cx - 14, cy), (cx + 14, cy), col, 2)
    cv2.line(frame, (cx, cy - 14), (cx, cy + 14), col, 2)
    cv2.arrowedLine(frame, fc, (cx, cy), col, 2, tipLength=0.2)

    cv2.line(frame, (fc[0] - 20, fc[1]), (fc[0] + 20, fc[1]), (255, 255, 255), 1)
    cv2.line(frame, (fc[0], fc[1] - 20), (fc[0], fc[1] + 20), (255, 255, 255), 1)

    cv2.putText(frame, f"{depth:.2f}m", (cx + 14, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)

    if grip:
        txt, c = "MAVLink: GRIP", (0, 255, 0)
    else:
        txt = f"MAVLink: vx={vx:+.2f} vy={vy:+.2f} vz={vz:+.2f} m/s"
        c   = (0, 200, 255)
    cv2.putText(frame, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, c, 2, cv2.LINE_AA)


# ─── MAIN ─────────────────────────────────────────────────

def main():
    print("Ładowanie modelu Hailo (.hef)...")
    target, ng, in_params, out_params = setup_hailo(MODEL_PATH)

    print("Uruchamianie RealSense...")
    pipeline, align, fx, fy = setup_realsense()

    fc_x, fc_y = RS_W // 2, RS_H // 2

    print("\n=== Live Feed — naciśnij Q aby wyjść ===\n")
    print(f"{'CZAS':10}  KOMENDA")
    print("-" * 65)

    last_print = 0.0

    try:
        # Kontekst InferVStreams — utrzymuje połączenie przez całą pętlę
        with InferVStreams(ng, in_params, out_params) as infer_pipeline:
            while True:
                # ── Pobierz klatki z RealSense ──────────────
                frames   = pipeline.wait_for_frames()
                aligned  = align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                frame = np.asanyarray(color_frame.get_data())   # (480, 640, 3) BGR

                # ── Preprocessing ───────────────────────────
                model_input = preprocess(frame, MODEL_W, MODEL_H)
                # Hailo oczekuje słownika {nazwa_warstwy: ndarray}
                input_data = {
                    list(in_params.keys())[0]: np.expand_dims(model_input, axis=0)
                }

                # ── Inferencja na Hailo-8L ──────────────────
                raw_outputs = infer_pipeline.infer(input_data)

                # ── Postprocessing ──────────────────────────
                dets = postprocess_yolov8(
                    raw_outputs, RS_W, RS_H, MODEL_W, MODEL_H, CONF_THRESHOLD
                )

                # Wybierz najbliższą piłkę (najmniejsza głębokość)
                best = None
                for x1, y1, x2, y2, conf in dets:
                    cx   = (x1 + x2) // 2
                    cy   = (y1 + y2) // 2
                    dep  = get_depth(depth_frame, cx, cy, RS_W, RS_H)
                    if best is None or dep < best[2]:
                        best = (cx, cy, dep, x1, y1, x2, y2)

                # ── Komendy + rysowanie ─────────────────────
                if best:
                    cx, cy, depth, x1, y1, x2, y2 = best
                    err_x_m = (cx - fc_x) * depth / fx
                    err_y_m = (cy - fc_y) * depth / fy

                    vx, vy, vz, grip = compute_mavlink_commands(err_x_m, err_y_m, depth)

                    now = time.time()
                    if now - last_print > 0.1:
                        print_mavlink(vx, vy, vz, grip, depth, err_x_m, err_y_m)
                        last_print = now

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 1)
                    draw_overlay(frame, cx, cy, RS_W, RS_H, depth, vx, vy, vz, grip)

                else:
                    now = time.time()
                    if now - last_print > 0.5:
                        print(f"[{time.strftime('%H:%M:%S')}]  NO BALL — HOVER  vx=0 vy=0 vz=0")
                        last_print = now

                    cv2.putText(frame, "NO BALL — HOVER", (10, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 80, 255), 2, cv2.LINE_AA)
                    cv2.line(frame, (fc_x - 20, fc_y), (fc_x + 20, fc_y), (255, 255, 255), 1)
                    cv2.line(frame, (fc_x, fc_y - 20), (fc_x, fc_y + 20), (255, 255, 255), 1)

                cv2.imshow("Ball Tracker [Hailo]", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        target.release()
        print("\nZatrzymano.")


if __name__ == "__main__":
    main()
