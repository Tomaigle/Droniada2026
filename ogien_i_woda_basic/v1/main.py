"""
Panel Anomaly Detector — detekcja kolorowych kartek na banerach PV
=================================================================
Założenia:
  - Baner (panel) to czarny prostokąt 200×100 cm (siatka 10×10 komórek 20×10 cm)
  - Układ współrzędnych: (1,1) = lewy DOLNY róg (biały panel narożny),
    X rośnie w prawo (1-10), Y rośnie w górę (1-10)
  - Kolorowe kartki: czerwona, niebieska, fioletowa, zielona, żółta, pomarańczowa
  - Skrypt wykrywa panele automatycznie (obroty 0°/45°/90°), a następnie
    raportuje anomalie w czasie rzeczywistym przez MQTT i lokalnie w konsoli.

Wymagania:
  pip install opencv-python numpy paho-mqtt

Użycie:
  # Obraz/wideo z pliku:
  python panel_detector.py --source ścieżka/do/pliku.jpg
  python panel_detector.py --source ścieżka/do/wideo.mp4

  # Kamera drona (live):
  python panel_detector.py --source 0          # kamera USB / domyślna
  python panel_detector.py --source rtsp://...  # strumień RTSP

  # MQTT (opcjonalne):
  python panel_detector.py --source 0 --mqtt-host 192.168.1.100

  # Podgląd okna:
  python panel_detector.py --source 0 --show
"""

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

# ── opcjonalne MQTT ──────────────────────────────────────────────────────────
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
# KONFIGURACJA
# ═══════════════════════════════════════════════════════════════════════════════

MQTT_TOPIC = "pv/anomalies"

# Zakresy kolorów w przestrzeni HSV (H: 0-179, S: 0-255, V: 0-255)
COLOR_RANGES = {
    "czerwona":    [(  0, 120,  80), ( 10, 255, 255),
                   (165, 120,  80), (179, 255, 255)],   # dwa zakresy dla czerwieni
    "pomarańczowa":[(11, 130,  80), ( 25, 255, 255)],
    "żółta":       [(26, 130,  80), ( 35, 255, 255)],
    "zielona":     [(36,  80,  60), ( 85, 255, 255)],
    "niebieska":   [(95,  80,  60), (130, 255, 255)],
    "fioletowa":   [(130, 60,  60), (160, 255, 255)],
}

# Minimalna powierzchnia konturu kartki (piksele²) — odfiltruje szumy
MIN_CARD_AREA = 300

# Minimalna powierzchnia banera (piksele²)
MIN_BANNER_AREA = 5000

# ═══════════════════════════════════════════════════════════════════════════════
# STRUKTURY DANYCH
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Anomaly:
    panel_id: int          # numer banera (1, 2, 3)
    color: str             # nazwa koloru
    grid_x: int            # współrzędna X siatki (1-10)
    grid_y: int            # współrzędna Y siatki (1-10)
    confidence: float      # pewność detekcji (0-1)
    pixel_center: tuple    # środek w pikselach (cx, cy) — do debugowania

    def to_dict(self):
        return {
            "panel_id": self.panel_id,
            "color": self.color,
            "grid_x": self.grid_x,
            "grid_y": self.grid_y,
            "confidence": round(self.confidence, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    def __str__(self):
        return (f"[Panel {self.panel_id}] {self.color:>12s}  "
                f"({self.grid_x:2d}, {self.grid_y:2d})  "
                f"pewność={self.confidence:.0%}")


@dataclass
class Panel:
    id: int
    contour: np.ndarray          # kontur banera w pikselach
    rect: tuple                  # cv2.minAreaRect → ((cx,cy),(w,h),angle)
    transform: np.ndarray        # macierz 3×3 do układu lokalnego banera
    width_px: float
    height_px: float
    anomalies: list = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# DETEKCJA BANERÓW
# ═══════════════════════════════════════════════════════════════════════════════


def detect_panels(frame: np.ndarray) -> list[Panel]:
    """
    Wykrywa czarne banery na obrazie.
    Zwraca listę obiektów Panel posortowanych lewym-górnym rogiem.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Czarne obszary: niskie V w HSV lub ciemne w szarości
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    black_mask = cv2.inRange(hsv, (0, 0, 0), (179, 255, 60))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    panels = []
    panel_id = 1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_BANNER_AREA:
            continue

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect

        # Upewnij się, że w ≥ h (baner leży poziomo lub jest obrócony)
        if w < h:
            w, h = h, w
            angle += 90

        # Macierz transformacji: piksel → układ lokalny banera (0..1 × 0..1)
        src_pts = cv2.boxPoints(rect).astype(np.float32)
        # Sortuj narożniki: lewy-dolny, prawy-dolny, prawy-górny, lewy-górny
        src_pts = _sort_corners(src_pts)
        dst_pts = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        panels.append(Panel(
            id=panel_id,
            contour=cnt,
            rect=rect,
            transform=M,
            width_px=w,
            height_px=h,
        ))
        panel_id += 1

    # Sortuj panele od lewej do prawej (po środku X)
    panels.sort(key=lambda p: p.rect[0][0])
    for i, p in enumerate(panels):
        p.id = i + 1

    return panels


def _sort_corners(pts: np.ndarray) -> np.ndarray:
    """Sortuje 4 narożniki w kolejności: lewy-dolny, prawy-dolny, prawy-górny, lewy-górny.

    Współrzędne w pikselach obrazu (Y rośnie w DÓŁ). Dla sumy s=x+y i różnicy d=y-x:
      - lewy-górny  (małe x, małe y)  → min s
      - prawy-dolny (duże x, duże y)  → max s
      - prawy-górny (duże x, małe y)  → min d
      - lewy-dolny  (małe x, duże y)  → max d
    Ta kolejność musi pasować do dst_pts=[[0,1],[1,1],[1,0],[0,0]] w detect_panels,
    czyli (lewy-dolny→(0,1)=siatka(1,1)), aby X,Y siatki nie były zamienione/odbite.
    """
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()   # = y - x
    return np.array([
        pts[np.argmax(d)],   # lewy-dolny
        pts[np.argmax(s)],   # prawy-dolny
        pts[np.argmin(d)],   # prawy-górny
        pts[np.argmin(s)],   # lewy-górny
    ], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# DETEKCJA KOLORÓW
# ═══════════════════════════════════════════════════════════════════════════════


def build_color_mask(hsv: np.ndarray, color_name: str) -> np.ndarray:
    """Tworzy maskę binarną dla danego koloru."""
    ranges = COLOR_RANGES[color_name]
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    # Obsługa wielu zakresów (np. czerwień zawijająca się na kole H)
    for i in range(0, len(ranges), 2):
        lo = np.array(ranges[i])
        hi = np.array(ranges[i + 1])
        mask |= cv2.inRange(hsv, lo, hi)
    return mask


def pixel_to_grid(px_norm: float, py_norm: float) -> tuple[int, int]:
    """
    Zamienia znormalizowane współrzędne na siatce (0..1 × 0..1)
    na współrzędne siatki (1..10 × 1..10).

    Układ lokalny banera:
      (0,0) = lewy-górny  →  siatka (1,10)
      (1,0) = prawy-górny →  siatka (10,10)
      (0,1) = lewy-dolny  →  siatka (1,1)   ← (1,1) = biały narożny panel
      (1,1) = prawy-dolny →  siatka (10,1)
    """
    grid_x = int(px_norm * 10) + 1
    grid_y = int((1 - py_norm) * 10) + 1
    grid_x = max(1, min(10, grid_x))
    grid_y = max(1, min(10, grid_y))
    return grid_x, grid_y


def detect_anomalies_on_panel(frame: np.ndarray,
                               panel: Panel) -> list[Anomaly]:
    """Wykrywa kolorowe kartki na konkretnym panelu."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_img, w_img = frame.shape[:2]

    anomalies = []
    seen_cells = {}  # (gx, gy) → najlepsza anomalia (unikalne kolory w komórce)

    for color_name in COLOR_RANGES:
        color_mask = build_color_mask(hsv, color_name)

        # Ogranicz do obszaru banera
        panel_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.drawContours(panel_mask, [panel.contour], -1, 255, -1)
        color_mask = cv2.bitwise_and(color_mask, panel_mask)

        # Morfologia — usuń szumy
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, k)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, k)

        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_CARD_AREA:
                continue

            M_cnt = cv2.moments(cnt)
            if M_cnt["m00"] == 0:
                continue
            cx = M_cnt["m10"] / M_cnt["m00"]
            cy = M_cnt["m01"] / M_cnt["m00"]

            # Transformuj piksel → układ lokalny banera
            pt = np.array([[[cx, cy]]], dtype=np.float32)
            pt_norm = cv2.perspectiveTransform(pt, panel.transform)[0][0]
            px_norm, py_norm = float(pt_norm[0]), float(pt_norm[1])

            if not (0 <= px_norm <= 1 and 0 <= py_norm <= 1):
                continue

            grid_x, grid_y = pixel_to_grid(px_norm, py_norm)

            # Pewność na podstawie stosunku powierzchni kartki do komórki
            cell_area_px = (panel.width_px * panel.height_px) / 100
            confidence = min(1.0, area / cell_area_px)

            anomaly = Anomaly(
                panel_id=panel.id,
                color=color_name,
                grid_x=grid_x,
                grid_y=grid_y,
                confidence=confidence,
                pixel_center=(int(cx), int(cy)),
            )

            key = (grid_x, grid_y)
            if key not in seen_cells or seen_cells[key].confidence < confidence:
                seen_cells[key] = anomaly

    return list(seen_cells.values())


# ═══════════════════════════════════════════════════════════════════════════════
# RAPORTOWANIE
# ═══════════════════════════════════════════════════════════════════════════════

class Reporter:
    def __init__(self, mqtt_host: Optional[str] = None,
                 mqtt_port: int = 1883):
        self.mqtt_client = None
        self._known: set[str] = set()  # już zgłoszone (panel, color, x, y)

        if mqtt_host and MQTT_AVAILABLE:
            self.mqtt_client = mqtt.Client()
            try:
                self.mqtt_client.connect(mqtt_host, mqtt_port, keepalive=60)
                self.mqtt_client.loop_start()
                print(f"[MQTT] Połączono z {mqtt_host}:{mqtt_port}")
            except Exception as e:
                print(f"[MQTT] Błąd połączenia: {e}")
                self.mqtt_client = None
        elif mqtt_host and not MQTT_AVAILABLE:
            print("[MQTT] Biblioteka paho-mqtt niedostępna — tylko konsola.")

    def report(self, anomalies: list[Anomaly]):
        for a in anomalies:
            key = f"{a.panel_id}:{a.color}:{a.grid_x}:{a.grid_y}"
            if key in self._known:
                continue  # nie duplikuj raportów
            self._known.add(key)

            print(a)  # konsola

            if self.mqtt_client:
                payload = json.dumps(a.to_dict())
                self.mqtt_client.publish(MQTT_TOPIC, payload)

    def reset(self):
        """Wyczyść pamięć — przydatne przy nowej misji."""
        self._known.clear()

    def stop(self):
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()


# ═══════════════════════════════════════════════════════════════════════════════
# WIZUALIZACJA (debug)
# ═══════════════════════════════════════════════════════════════════════════════

COLOR_BGR = {
    "czerwona":     (0,   0, 220),
    "pomarańczowa": (0, 140, 255),
    "żółta":        (0, 220, 220),
    "zielona":      (0, 200,   0),
    "niebieska":    (200,  0,   0),
    "fioletowa":    (180,  0, 180),
}


def draw_debug(frame: np.ndarray,
               panels: list[Panel],
               anomalies: list[Anomaly]) -> np.ndarray:
    out = frame.copy()

    # Rysuj kontury banerów
    for panel in panels:
        box = cv2.boxPoints(panel.rect).astype(int)
        cv2.drawContours(out, [box], -1, (255, 255, 255), 2)
        cx, cy = int(panel.rect[0][0]), int(panel.rect[0][1])
        cv2.putText(out, f"Panel {panel.id}", (cx - 30, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Rysuj anomalie
    for a in anomalies:
        bgr = COLOR_BGR.get(a.color, (200, 200, 200))
        cx, cy = a.pixel_center
        cv2.circle(out, (cx, cy), 12, bgr, -1)
        cv2.circle(out, (cx, cy), 12, (255, 255, 255), 2)
        label = f"({a.grid_x},{a.grid_y})"
        cv2.putText(out, label, (cx + 14, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# GŁÓWNA PĘTLA
# ═══════════════════════════════════════════════════════════════════════════════


def run(source, show: bool, mqtt_host: Optional[str], mqtt_port: int):
    # Otwórz źródło wideo / kamerę
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        print(f"[ERROR] Nie można otworzyć źródła: {source}")
        sys.exit(1)

    reporter = Reporter(mqtt_host=mqtt_host, mqtt_port=mqtt_port)
    print("=" * 60)
    print("  Panel Anomaly Detector — start")
    print("  Naciśnij Q aby wyjść, R aby zresetować raporty")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            # Koniec pliku lub błąd kamery
            break

        panels = detect_panels(frame)
        all_anomalies: list[Anomaly] = []

        for panel in panels:
            found = detect_anomalies_on_panel(frame, panel)
            panel.anomalies = found
            all_anomalies.extend(found)

        reporter.report(all_anomalies)

        if show:
            debug_frame = draw_debug(frame, panels, all_anomalies)
            cv2.imshow("Panel Detector", debug_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                reporter.reset()
                print("[INFO] Pamięć raportów wyczyszczona.")

        # Dla obrazów statycznych: jeden cykl i koniec
        if not cap.get(cv2.CAP_PROP_FRAME_COUNT) > 1:
            time.sleep(0.03)  # ~30 fps throttle dla strumieni

    cap.release()
    if show:
        cv2.destroyAllWindows()
    reporter.stop()
    print("[INFO] Zakończono.")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Detekcja anomalii na panelach PV (zawody dronowe)")
    parser.add_argument("--source", default="0",
        help="Źródło obrazu: ścieżka do pliku, RTSP URL lub numer kamery (domyślnie 0)")
    parser.add_argument("--show", action="store_true",
        help="Pokaż okno podglądu z adnotacjami")
    parser.add_argument("--mqtt-host", default=None,
        help="Adres IP brokera MQTT (opcjonalne)")
    parser.add_argument("--mqtt-port", type=int, default=1883,
        help="Port brokera MQTT (domyślnie 1883)")
    args = parser.parse_args()

    run(args.source, args.show, args.mqtt_host, args.mqtt_port)


if __name__ == "__main__":
    main()
