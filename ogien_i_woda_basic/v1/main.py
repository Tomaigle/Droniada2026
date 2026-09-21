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
    "pomarańczowa":[( 8, 130,  80), ( 19, 255, 255)],
    "żółta":       [(20, 100,  80), ( 35, 255, 255)],
    "zielona":     [(36,  80,  60), ( 85, 255, 255)],
    "niebieska":   [(95,  80,  60), (130, 255, 255)],
    "fioletowa":   [(130, 60,  60), (160, 255, 255)],
}

# Minimalna powierzchnia konturu kartki (piksele²) — odfiltruje szumy.
# Kartki na obróconych / dalekich panelach po rzutowaniu bywają małe (~150-300 px²).
MIN_CARD_AREA = 140

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
    quad: Optional[np.ndarray] = None   # 4 narożniki (perspektywa) — dokładniejsze niż rect


# ═══════════════════════════════════════════════════════════════════════════════
# DETEKCJA BANERÓW
# ═══════════════════════════════════════════════════════════════════════════════


# Panel w dziennym świetle to nie czerń: szkło odbija niebo → niebieskawa szarość
# (zdjęcia z ręki: H~115, S~30, V 50-140). Niebo jest jaśniejsze (V>190) i odpada.
PANEL_GRAY_LO = (104, 4, 20)
PANEL_GRAY_HI = (124, 50, 165)

# Minimalne wypełnienie prostokąta otaczającego przez wypukłą otoczkę kandydata.
MIN_PANEL_RECTFILL = 0.70


def _content_roi(hsv: np.ndarray) -> tuple:
    """Bounding box treści bez czarnych pasów (pillarbox/letterbox) → (x0, y0, x1, y1)."""
    v = hsv[:, :, 2]
    cols = np.where(v.max(axis=0) > 6)[0]
    rows = np.where(v.max(axis=1) > 6)[0]
    if cols.size == 0 or rows.size == 0:
        return 0, 0, v.shape[1], v.shape[0]
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def _line_intersect(p1, p2, p3, p4) -> Optional[np.ndarray]:
    """Przecięcie prostych p1-p2 i p3-p4 (None gdy prawie równoległe)."""
    d1, d2 = p2 - p1, p4 - p3
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-6 * (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-9):
        return None
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / den
    return p1 + t * d1


def _hull_quad(hull: np.ndarray) -> Optional[np.ndarray]:
    """Wypukła otoczka → czworokąt z zachowaną perspektywą.

    Biały narożnik (1,1) i kartki nie wchodzą do maski banera, więc otoczka ma ścięty
    róg (5+ wierzchołków). Usuwamy najkrótszą krawędź i przedłużamy jej sąsiadki aż
    do przecięcia — odzyskuje to prawdziwy narożnik zamiast go ucinać.
    """
    peri = cv2.arcLength(hull, True)
    poly = cv2.approxPolyDP(hull, 0.008 * peri, True).reshape(-1, 2).astype(np.float64)
    while len(poly) > 4:
        n = len(poly)
        lens = [np.linalg.norm(poly[(i + 1) % n] - poly[i]) for i in range(n)]
        i = int(np.argmin(lens))
        pt = _line_intersect(poly[i - 1], poly[i], poly[(i + 1) % n], poly[(i + 2) % n])
        if pt is None:
            return None
        poly = np.array([pt if j == i else poly[j] for j in range(n)
                         if j != (i + 1) % n])
    if len(poly) != 4:
        return None
    quad = poly.astype(np.float32)
    ha, qa = cv2.contourArea(hull), cv2.contourArea(quad)
    if ha <= 0 or not (0.85 < qa / ha < 1.25) or not cv2.isContourConvex(quad.reshape(-1, 1, 2)):
        return None   # wyciek maski wypaczył otoczkę → użyj minAreaRect
    return quad


def _refine_quad(pts: np.ndarray, quad: np.ndarray, bands=(0.12, 0.06, 0.03, 0.02)) -> np.ndarray:
    """Dopasuj każdą z 4 krawędzi odporną regresją (Huber) do punktów konturu leżących
    blisko niej. Wycieki maski (samochód, drzewo przy narożniku) psują wierzchołki
    otoczki, ale nie proste wzdłuż długich boków."""
    q = quad.astype(np.float64)
    pts = pts.astype(np.float64)
    for band in bands:
        lines = []
        for i in range(4):
            a, b = q[i], q[(i + 1) % 4]
            d = b - a
            L = np.linalg.norm(d)
            if L < 1e-6:
                return quad
            u = d / L
            n = np.array([-u[1], u[0]])
            rel = pts - a
            t = rel @ u / L
            dist = np.abs(rel @ n)
            sel = pts[(dist < max(6.0, band * L)) & (t > 0.05) & (t < 0.95)]
            if len(sel) < 12:
                return quad
            vx, vy, x0, y0 = cv2.fitLine(sel.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            lines.append((np.array([x0, y0]), np.array([vx, vy])))
        new = []
        for i in range(4):
            p1, d1 = lines[i - 1]
            p2, d2 = lines[i]
            pt = _line_intersect(p1, p1 + d1, p2, p2 + d2)
            if pt is None:
                return quad
            new.append(pt)
        q = np.array(new)
    out = q.astype(np.float32)
    a0, a1 = cv2.contourArea(quad.astype(np.float32)), cv2.contourArea(out)
    if a0 <= 0 or not (0.8 < a1 / a0 < 1.25) or not cv2.isContourConvex(out.reshape(-1, 1, 2)):
        return quad   # dopasowanie uciekło (wyciek maski) → zostaw czworokąt z otoczki
    return out


# Zestawy (zamknięcie px, otwarcie jako ułamek wysokości kadru, dolna granica odcienia H).
# Wyższe H_min odcina metaliczne auta (H~101), ale gubi jaśniejsze fragmenty banera. Jeden zestaw nie
# pasuje do każdego zdjęcia: małe zamknięcie zostawia baner podzielony białą siatką/odblaskiem,
# duże zlewa go z tłem. Próbujemy kilku i wybieramy kandydata najlepiej dopasowanego.
PANEL_MASK_CONFIGS = ((11, 0.04, 104), (17, 0.04, 104), (11, 0.06, 104), (17, 0.06, 104),
                      (11, 0.06, 108), (15, 0.06, 108), (11, 0.09, 104), (15, 0.09, 108))
MIN_QUAD_FIT = 0.85       # IoU(czworokąt, wypełniona otoczka) – poniżej: wyciek maski
MIN_PANEL_DARK_FRAC = 0.60  # min. udział ciemnych/szarych pikseli we wnętrzu czworokąta
MIN_EDGE_SUPPORT = 0.75   # ułamek punktów konturu leżących przy bokach czworokąta


def _make_mask(hsv: np.ndarray, roi: np.ndarray, close_px: int, k_open: int,
               h_min: int = PANEL_GRAY_LO[0]) -> np.ndarray:
    # Ciemne i mało nasycone (render / cień / szkło w słabym świetle) LUB niebieskawo-szare.
    # Ograniczenie nasycenia odcina ciemną zieleń drzew, które zlewały się z banerem.
    dark = cv2.inRange(hsv, (0, 0, 0), (179, 70, 70))
    gray = cv2.inRange(hsv, (h_min,) + PANEL_GRAY_LO[1:], PANEL_GRAY_HI)
    rect7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dark = cv2.morphologyEx(cv2.bitwise_and(dark, roi), cv2.MORPH_CLOSE, rect7)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, rect7)

    # Niebieskawo-szary jest mniej selektywny (drzewa, samochody, cienie) → mocniejsze
    # otwarcie odcina cienkie „mostki" łączące baner z tłem.
    gray = cv2.morphologyEx(cv2.bitwise_and(gray, roi), cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (close_px, close_px)))
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (k_open, k_open)))
    mask = cv2.bitwise_or(dark, gray)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))


def _candidate_from_contour(frame: np.ndarray, cnt: np.ndarray):
    """Kontur → (Panel, fit, support) albo None. fit = IoU(czworokąt, otoczka)."""
    if cv2.contourArea(cnt) < MIN_BANNER_AREA:
        return None
    hull = cv2.convexHull(cnt)
    rect = cv2.minAreaRect(hull)
    (_, _), (w, h), _ = rect
    hull_area = cv2.contourArea(hull)
    if w * h <= 0 or hull_area / (w * h) < MIN_PANEL_RECTFILL:
        return None  # nie prostokąt (trawa, cień, chmura...)
    if w < h:
        w, h = h, w

    # Czworokąt z perspektywą; fallback: prostokąt minAreaRect
    box = _hull_quad(hull)
    if box is None:
        box = cv2.boxPoints(rect).astype(np.float32)
    box = _refine_quad(cnt.reshape(-1, 2), box)

    # Najpierw ułóż narożniki wg białego znacznika (1,1) + długości boków — działa dla
    # DOWOLNEGO obrotu. Bez znacznika to nie baner (drzewo, cień, okna, auto).
    src_pts = _order_corners_from_marker(frame, box)
    if src_pts is None:
        return None
    dst_pts = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    x, y, bw, bh = cv2.boundingRect(np.vstack([hull.reshape(-1, 2), box]).astype(np.int32))
    m_q = np.zeros((bh + 1, bw + 1), np.uint8)
    m_h = np.zeros_like(m_q)
    cv2.fillConvexPoly(m_q, (box - (x, y)).astype(np.int32), 1)
    cv2.fillConvexPoly(m_h, (hull.reshape(-1, 2) - (x, y)).astype(np.int32), 1)
    union = int(np.count_nonzero(m_q | m_h))
    fit = np.count_nonzero(m_q & m_h) / union if union else 0.0

    # Wnętrze banera jest ciemne/szare (jasne są tylko linie siatki, kartki, znacznik).
    # Jasne wnętrze (białe auto, tynk, niebo) → nie baner, mimo białego „znacznika" w rogu.
    roi_hsv = cv2.cvtColor(frame[y:y + m_q.shape[0], x:x + m_q.shape[1]], cv2.COLOR_BGR2HSV)
    inside = m_q[:roi_hsv.shape[0], :roi_hsv.shape[1]] > 0
    if inside.any():
        darkish = (roi_hsv[:, :, 2] <= 175) & (roi_hsv[:, :, 1] <= 70)
        if float(np.mean(darkish[inside])) < MIN_PANEL_DARK_FRAC:
            return None

    # Poparcie krawędzi: prawdziwy baner ma prosty obrys; wyciek (auto, płot) go łamie.
    pts = cnt.reshape(-1, 2).astype(np.float32)
    tol = max(5.0, 0.02 * cv2.arcLength(box.reshape(-1, 1, 2), True))
    dists = []
    for i in range(4):
        e = box[(i + 1) % 4] - box[i]
        rel = pts - box[i]
        dists.append(np.abs(e[0] * rel[:, 1] - e[1] * rel[:, 0]) / max(1e-6, float(np.linalg.norm(e))))
    d = np.min(dists, axis=0)
    support = float(np.mean(d < tol))

    panel = Panel(id=0, contour=cnt, rect=rect, transform=M,
                  width_px=w, height_px=h, quad=box)
    return panel, fit, support


def detect_panels(frame: np.ndarray) -> list[Panel]:
    """
    Wykrywa banery na obrazie: czarne (render / ostre światło) lub niebieskawo-szare
    (zdjęcia w dziennym świetle). Czarne pasy wokół kadru są ignorowane.
    Zwraca listę obiektów Panel posortowanych od lewej do prawej.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    x0, y0, x1, y1 = _content_roi(hsv)
    roi = np.zeros(hsv.shape[:2], dtype=np.uint8)
    roi[y0:y1, x0:x1] = 255

    cands = []   # (Panel, fit, support)
    for close_px, open_frac, h_min in PANEL_MASK_CONFIGS:
        k_open = max(7, int(open_frac * (y1 - y0)) | 1)
        mask = _make_mask(hsv, roi, close_px, k_open, h_min)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            c = _candidate_from_contour(frame, cnt)
            if c is not None:
                cands.append(c)

    # Ten sam baner wychodzi z kilku konfiguracji: zostaw jednego. Preferuj dobrze
    # dopasowane (bez wycieku), a wśród nich największy (nie fragment).
    cands.sort(key=lambda c: (c[1] >= MIN_QUAD_FIT and c[2] >= MIN_EDGE_SUPPORT,
                              cv2.contourArea(c[0].quad), c[2]), reverse=True)
    panels: list[Panel] = []
    for panel, _fit, _support in cands:
        cx, cy = panel.rect[0]
        if any(cv2.pointPolygonTest(k.quad.astype(np.float32), (float(cx), float(cy)), False) >= 0
               or cv2.pointPolygonTest(panel.quad.astype(np.float32),
                                       (float(k.rect[0][0]), float(k.rect[0][1])), False) >= 0
               for k in panels):
            continue
        panels.append(panel)

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
    Używane TYLKO jako fallback, gdy biały znacznik (1,1) jest niepewny
    (_order_corners_from_marker zwróciło None). Dla paneli ~45° ta heurystyka bywa
    zdegenerowana; wtedy orientacja może być błędna — normalnie ścieżka ze znacznikiem
    to pokrywa. Kolejność pasuje do dst_pts=[[0,1],[1,1],[1,0],[0,0]] w detect_panels.
    """
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()   # = y - x
    return np.array([
        pts[np.argmax(d)],   # lewy-dolny
        pts[np.argmax(s)],   # prawy-dolny
        pts[np.argmin(d)],   # prawy-górny
        pts[np.argmin(s)],   # lewy-górny
    ], dtype=np.float32)


def _corner_whiteness(frame: np.ndarray, corner_pt: np.ndarray,
                      centroid: np.ndarray) -> float:
    """Fraction of a patch INSIDE the panel at this corner that is white-ish.

    The white (1,1) marker fills the corner cell. Sampling exactly at the corner
    vertex catches the black border / the grey background, so step ~18% of the way
    from the corner toward the panel centroid (roughly the centre of the corner cell)
    and sample there.
    """
    try:
        h, w = frame.shape[:2]
        sample = corner_pt + 0.18 * (centroid - corner_pt)
        sx, sy = int(round(sample[0])), int(round(sample[1]))
        r = 6
        x0, x1 = max(0, sx - r), min(w, sx + r)
        y0, y1 = max(0, sy - r), min(h, sy + r)
        patch = frame[y0:y1, x0:x1]
        if patch.size == 0:
            return 0.0
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        white = (hsv[:, :, 2] > 150) & (hsv[:, :, 1] < 60)
        return float(np.mean(white))
    except Exception:
        return 0.0


def _marker_cell_ok(frame: np.ndarray, corner: np.ndarray, quad: np.ndarray) -> bool:
    """Czy jasna plama w narożniku ma rozmiar JEDNEJ komórki (~1% banera)?

    Odrzuca narożniki graniczące z niebem / białym autem / tynkiem: tam jasny obszar
    ciągnie się daleko poza komórkę (zalewa okno albo dotyka jego brzegu)."""
    try:
        h, w = frame.shape[:2]
        centroid = quad.mean(axis=0)
        sample = corner + 0.10 * (centroid - corner)   # środek komórki (1/20 boku w obu osiach)
        sx, sy = int(round(sample[0])), int(round(sample[1]))
        span = 0.4 * max(np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4))
        x0, x1 = max(0, int(sx - span)), min(w, int(sx + span))
        y0, y1 = max(0, int(sy - span), ), min(h, int(sy + span))
        if not (x0 < sx < x1 and y0 < sy < y1):
            return False
        hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        vs = float(np.median(hsv[max(0, sy - y0 - 3):sy - y0 + 4,
                                 max(0, sx - x0 - 3):sx - x0 + 4, 2]))
        bright = ((hsv[:, :, 2] > max(90.0, 0.75 * vs)) & (hsv[:, :, 1] < 70)).astype(np.uint8)
        # Biała siatka i ramka łączą się z komórką znacznika → otwarcie o skali komórki
        # (~1/10 krótszego boku) odcina cienkie linie, zostawia sam prostokąt.
        min_side = min(np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4))
        k = max(3, int(0.3 * min_side / 10) | 1)
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        flood = bright.copy()
        ff = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), np.uint8)
        cv2.floodFill(flood, ff, (sx - x0, sy - y0), 2)
        comp = flood == 2
        if not comp.any():
            return False
        ys, xs = np.where(comp)
        if xs.min() == 0 or ys.min() == 0 or xs.max() == comp.shape[1] - 1 or ys.max() == comp.shape[0] - 1:
            return False                       # dotyka brzegu okna → obszar za duży
        quad_area = cv2.contourArea(quad.astype(np.float32))
        return comp.sum() <= 0.05 * quad_area  # komórka ≈ 1% banera
    except Exception:
        return False


def _rotate_corners_for_white_marker(frame: np.ndarray, src_pts: np.ndarray) -> np.ndarray:
    """Cyclically rotate the (BL,BR,TR,TL)-ordered corner list so the white (1,1)
    marker corner sits at index 0 (which dst_pts maps to grid (1,1)).

    Falls back to the geometric ordering when no corner is clearly whiter than the
    others (real photo, marker occluded, etc.)."""
    centroid = src_pts.mean(axis=0)
    scores = [_corner_whiteness(frame, c, centroid) for c in src_pts]
    best = int(np.argmax(scores))
    # require a clear signal AND that it beats the runner-up
    ranked = sorted(scores, reverse=True)
    if ranked[0] < 0.3 or ranked[0] < ranked[1] + 0.15:
        return src_pts
    if best == 0:
        return src_pts
    return np.array([src_pts[(best + i) % 4] for i in range(4)], dtype=np.float32)


def _order_corners_from_marker(frame: np.ndarray, box: np.ndarray):
    """Ułóż 4 narożniki minAreaRect jako (BL, BR, TR, TL) używając białego znacznika (1,1)
    i długości boków. Odporne na obrót (0°/45°/90°). Zwraca None, gdy znacznik niepewny.

    BL  = najbielszy narożnik (siatka (1,1))
    TR  = narożnik po przekątnej (przeciwległy do BL w kolejności cyklicznej)
    BR  = z dwóch sąsiednich ten wzdłuż DŁUŻSZEGO boku (oś X, 2 m)
    TL  = z dwóch sąsiednich ten wzdłuż KRÓTSZEGO boku (oś Y, 1 m)
    """
    c = box.mean(axis=0)
    scores = [_corner_whiteness(frame, p, c) for p in box]
    ranked = sorted(scores, reverse=True)
    if ranked[0] < 0.3 or ranked[0] < ranked[1] + 0.15:
        return None
    bl = int(np.argmax(scores))
    if not _marker_cell_ok(frame, box[bl], box):
        return None
    # Narożniki są w kolejności cyklicznej → przeciwległy to (bl+2). Nie „najdalszy": przy
    # perspektywie długi bok bywa dłuższy niż przekątna.
    tr = (bl + 2) % 4
    adj = sorted(((bl + 1) % 4, (bl + 3) % 4),
                 key=lambda i: float(np.linalg.norm(box[i] - box[bl])), reverse=True)
    br, tl = adj[0], adj[1]
    return np.array([box[bl], box[br], box[tr], box[tl]], dtype=np.float32)


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

    # Maska banera: wypełniony prostokąt minAreaRect (nie surowy kontur, który po
    # morfologii i przez biały znacznik (1,1) bywa „zjedzony” przy krawędziach i gubi
    # kartki w skrajnych kolumnach/wierszach). Lekko dylatowana dla zapasu.
    panel_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    box = (panel.quad if panel.quad is not None
           else cv2.boxPoints(panel.rect)).astype(np.int32)
    cv2.fillConvexPoly(panel_mask, box, 255)
    panel_mask = cv2.dilate(panel_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))

    for color_name in COLOR_RANGES:
        color_mask = build_color_mask(hsv, color_name)
        color_mask = cv2.bitwise_and(color_mask, panel_mask)

        # Morfologia — usuń szumy (mały kernel, by nie skasować drobnych kartek)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, k)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, k)

        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_CARD_AREA:
                continue

            # Kartka ≈ jedna komórka: odrzuć zbyt duże / poszarpane plamy (trawa, odblaski)
            cell_area_px = (panel.width_px * panel.height_px) / 100
            if area > 3.0 * cell_area_px:
                continue
            (_, (rw, rh), _) = cv2.minAreaRect(cnt)
            if rw * rh <= 0 or area / (rw * rh) < 0.6:
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

            # mały margines: środek skrajnej komórki potrafi wypaść tuż za [0,1]
            if not (-0.04 <= px_norm <= 1.04 and -0.04 <= py_norm <= 1.04):
                continue
            px_norm = min(1.0, max(0.0, px_norm))
            py_norm = min(1.0, max(0.0, py_norm))

            grid_x, grid_y = pixel_to_grid(px_norm, py_norm)

            # Pewność na podstawie stosunku powierzchni kartki do komórki
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
        box = (panel.quad if panel.quad is not None
               else cv2.boxPoints(panel.rect)).astype(int)
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
