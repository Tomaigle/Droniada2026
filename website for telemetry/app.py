"""
app.py — RealSense + YOLO → Flask MJPEG stream + telemetry API.

Usage:
    python app.py --model path/to/model.pt [--port 5000] [--conf 0.45] [--sim]

Args:
    --model     Path to YOLO .pt weights (required unless --sim)
    --port      Flask port (default 5000)
    --conf      YOLO confidence threshold (default 0.45)
    --classes   Comma-separated class IDs to filter (e.g. 0,1,2). Default: all
    --sim       Simulate camera with webcam (no RealSense required)
    --no-hud    Disable on-stream HUD overlay

Browse: http://<host-ip>:<port>
"""

import argparse
import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Shared state ───────────────────────────────────────────────────────────────

_lock = threading.Lock()
_latest_frame: np.ndarray | None = None
_telemetry: dict = {
    "fps": 0.0,
    "detections": [],
    "model": "",
    "resolution": "",
    "depth_available": False,
}

# ── Overlay / HUD ──────────────────────────────────────────────────────────────

COLOUR_MAP = {
    "blue": (255, 100, 30),
    "red": (30, 30, 220),
    "yellow": (0, 200, 220),
    "barrel": (0, 165, 255),
    "unknown": (180, 180, 180),
}


def _colour_for(label: str) -> tuple:
    for k, v in COLOUR_MAP.items():
        if k in label.lower():
            return v
    return (180, 180, 180)


def draw_hud(
    frame: np.ndarray, detections: list[dict], fps: float, depth_ok: bool
) -> np.ndarray:
    """Draw bounding boxes, labels, depth bars, and HUD panel onto frame."""
    out = frame.copy()
    h, w = out.shape[:2]

    # ── Detections ─────────────────────────────────────────────────────────────
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        label = d["label"]
        conf = d["conf"]
        depth = d.get("depth", 0.0)
        col = _colour_for(label)

        # Box
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)

        # Label pill background
        tag = f"{label} {conf:.2f}"
        if depth > 0:
            tag += f"  {depth:.2f}m"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 8, y1), col, -1)
        cv2.putText(
            out,
            tag,
            (x1 + 4, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Centre dot
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(out, (cx, cy), 4, col, -1)

        # Depth bar (right edge of bbox)
        if depth > 0 and depth < 5.0:
            bar_h = int((1 - depth / 5.0) * (y2 - y1))
            cv2.rectangle(out, (x2 + 4, y2 - bar_h), (x2 + 10, y2), col, -1)

    # ── HUD panel (top-left) ───────────────────────────────────────────────────
    panel_lines = [
        f"FPS: {fps:.1f}",
        f"Detections: {len(detections)}",
        f"Depth: {'OK' if depth_ok else 'N/A'}",
    ]
    pad = 8
    line_h = 18
    panel_h = pad * 2 + line_h * len(panel_lines)
    panel_w = 160
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, out, 0.5, 0, out)
    for i, line in enumerate(panel_lines):
        cv2.putText(
            out,
            line,
            (pad, pad + (i + 1) * line_h - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 210, 255),
            1,
            cv2.LINE_AA,
        )

    # ── Crosshair ─────────────────────────────────────────────────────────────
    cx2, cy2 = w // 2, h // 2
    cv2.line(out, (cx2 - 15, cy2), (cx2 + 15, cy2), (0, 210, 255), 1)
    cv2.line(out, (cx2, cy2 - 15), (cx2, cy2 + 15), (0, 210, 255), 1)
    cv2.circle(out, (cx2, cy2), 20, (0, 210, 255), 1)

    return out


# ── Camera + YOLO thread ───────────────────────────────────────────────────────


def camera_loop(args) -> None:
    global _latest_frame, _telemetry

    # ── Model ──────────────────────────────────────────────────────────────────
    from ultralytics import YOLO

    model_path = "best.pt"
    log.info("Loading YOLO: %s", model_path)
    model = YOLO(model_path)
    class_names = model.names  # {id: name}

    filter_classes = None
    if args.classes:
        filter_classes = [int(c) for c in args.classes.split(",")]
        log.info("Filtering classes: %s", filter_classes)

    conf_thresh = args.conf

    # ── Camera ─────────────────────────────────────────────────────────────────
    depth_available = False
    if args.sim:
        log.info("SIM mode — using webcam 0")
        cap = cv2.VideoCapture(0)

        def get_frame():
            ok, f = cap.read()
            return (f if ok else None), None
    else:
        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = pipeline.start(cfg)
        align = rs.align(rs.stream.color)
        intr = rs.video_stream_profile(
            profile.get_stream(rs.stream.depth)
        ).get_intrinsics()
        fx, fy = intr.fx, intr.fy
        depth_available = True
        log.info("RealSense ready — fx=%.1f fy=%.1f", fx, fy)

        def get_frame():
            try:
                frames = pipeline.wait_for_frames(timeout_ms=2000)
                aligned = align.process(frames)
                cf = aligned.get_color_frame()
                df = aligned.get_depth_frame()
                if not cf or not df:
                    return None, None
                return np.asanyarray(cf.get_data()), df
            except RuntimeError as e:
                log.warning("Frame grab: %s", e)
                return None, None

        def sample_depth(depth_frame, cx, cy, w=640, h=480) -> float:
            samples = []
            for dx in (-3, 0, 3):
                for dy in (-3, 0, 3):
                    x = int(np.clip(cx + dx, 0, w - 1))
                    y = int(np.clip(cy + dy, 0, h - 1))
                    d = depth_frame.get_distance(x, y)
                    if d > 0.05:
                        samples.append(d)
            return float(np.median(samples)) if samples else 0.0

    # ── Loop ───────────────────────────────────────────────────────────────────
    fps_counter, t0 = 0, time.time()
    fps_display = 0.0

    while True:
        frame, depth_frame = get_frame()
        if frame is None:
            time.sleep(0.02)
            continue

        h_f, w_f = frame.shape[:2]

        # YOLO inference
        results = model(
            frame,
            verbose=False,
            classes=filter_classes,
            conf=conf_thresh,
        )[0]

        detections = []
        for box in results.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cls_id = int(box.cls[0])
            label = class_names.get(cls_id, str(cls_id))

            depth = 0.0
            if depth_available and depth_frame is not None:
                depth = sample_depth(depth_frame, cx, cy, w_f, h_f)

            detections.append(
                {
                    "label": label,
                    "conf": round(conf, 3),
                    "bbox": [x1, y1, x2, y2],
                    "cx": cx,
                    "cy": cy,
                    "depth": round(depth, 3),
                }
            )

        # FPS
        fps_counter += 1
        if time.time() - t0 >= 1.0:
            fps_display = fps_counter / (time.time() - t0)
            fps_counter, t0 = 0, time.time()

        # HUD
        if not args.no_hud:
            annotated = draw_hud(frame, detections, fps_display, depth_available)
        else:
            annotated = frame

        # Publish
        with _lock:
            _latest_frame = annotated.copy()
            _telemetry = {
                "fps": round(fps_display, 1),
                "detections": detections,
                "model": Path(model_path).name,
                "resolution": f"{w_f}×{h_f}",
                "depth_available": depth_available,
                "det_count": len(detections),
            }

    if args.sim:
        cap.release()
    else:
        pipeline.stop()


# ── Flask ──────────────────────────────────────────────────────────────────────

app = Flask(__name__)


def _mjpeg_gen():
    while True:
        with _lock:
            frame = _latest_frame
        if frame is None:
            time.sleep(0.03)
            continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(1 / 20)  # 20 fps cap


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(_mjpeg_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/telemetry")
def telemetry():
    with _lock:
        data = dict(_telemetry)
    return jsonify(data)


# ── Entry ──────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="RealSense + YOLO → Flask stream")
    p.add_argument("--port", type=int, default=69420)
    p.add_argument("--conf", type=float, default=0.45, help="YOLO conf threshold")
    p.add_argument("--classes", default="", help="Comma-sep class IDs to filter")
    p.add_argument("--sim", action="store_true", help="Use webcam instead of RealSense")
    p.add_argument("--no-hud", action="store_true", help="Disable HUD overlay")
    return p.parse_args()


def main():
    args = parse_args()

    t = threading.Thread(target=camera_loop, args=(args,), daemon=True, name="cam-yolo")
    t.start()
    log.info("Camera thread started")
    log.info("Dashboard → http://0.0.0.0:%d", args.port)

    app.run(
        host="0.0.0.0", port=args.port, debug=False, use_reloader=False, threaded=True
    )


if __name__ == "__main__":
    main()
