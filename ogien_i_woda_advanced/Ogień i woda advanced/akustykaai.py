"""
Klasyfikator dźwięków bojowych — DSP Advanced
==============================================
Wymagania:  pip install librosa sounddevice numpy scipy
Python:     3.8+ (w tym 3.14), Windows/Linux/macOS

Uruchomienie (mikrofon na żywo):
    python sound_classifier.py

Uruchomienie na pliku WAV (testy):
    python sound_classifier.py --file próbka.wav

Lista urządzeń audio:
    python sound_classifier.py --list-devices
    python sound_classifier.py --device 2
"""

import argparse
import json
import time
import threading
import queue
from collections import defaultdict
from datetime import datetime

import numpy as np
import librosa
import sounddevice as sd


# ══════════════════════════════════════════════════════════════
# PARAMETRY
# ══════════════════════════════════════════════════════════════
SAMPLE_RATE    = 16000
WINDOW_SEC     = 2.0
OVERLAP_SEC    = 1.0
SILENCE_RMS    = 0.008
MIN_CONFIDENCE = 0.35

WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)
HOP_SAMPLES    = int(SAMPLE_RATE * (WINDOW_SEC - OVERLAP_SEC))

# ══════════════════════════════════════════════════════════════
# KATEGORIE
# ══════════════════════════════════════════════════════════════
CATEGORIES = [
    "woda_wyciek",
    "pozar_trzask",
    "wystrzal_krab",
    "bomba_lotnicza",
    "kolumna_pancerna",
    "serie_bron_maszynowej",
]

CONTINUOUS = {"woda_wyciek", "pozar_trzask", "kolumna_pancerna"}

LABELS_PL = {
    "woda_wyciek":           "Woda (wyciek z rury)",
    "pozar_trzask":          "Pozar (trzask poszycia)",
    "wystrzal_krab":         "Wystrzal Krab",
    "bomba_lotnicza":        "Bomba lotnicza",
    "kolumna_pancerna":      "Kolumna pancerna",
    "serie_bron_maszynowej": "Serie z broni maszynowej",
}


# ══════════════════════════════════════════════════════════════
# EKSTRAKCJA CECH
# ══════════════════════════════════════════════════════════════

def extract_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    rms      = float(np.sqrt(np.mean(audio ** 2)))
    zcr      = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
    sc       = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
    rolloff  = float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=audio)))

    stft     = np.abs(librosa.stft(audio))
    freqs    = librosa.fft_frequencies(sr=sr)
    low_mask = freqs < 500
    low_energy = float(stft[low_mask].sum() / (stft.sum() + 1e-9))

    mu       = np.mean(audio)
    sigma    = np.std(audio) + 1e-9
    kurtosis = float(np.mean(((audio - mu) / sigma) ** 4))

    return {
        "rms": rms, "zcr": zcr, "sc": sc,
        "rolloff": rolloff, "flatness": flatness,
        "low_energy": low_energy, "kurtosis": kurtosis,
    }


# ══════════════════════════════════════════════════════════════
# KLASYFIKATOR
# ══════════════════════════════════════════════════════════════

def classify(features: dict) -> tuple:
    """
    Ręczne drzewo decyzyjne.

    Kluczowe cechy i ich znaczenie:
      kurtosis  > 8   = sygnał impulsowy (strzały, wybuchy)
      zcr       > 0.12 = szybkie impulsy (seria z broni)
      low_energy > 0.55 = dominuje bas (bomba > Krab)
      sc        < 1200 = ciemny dźwięk (silnik)
      flatness  > 0.2  = szum (woda)
      sc 1000-3500     = srednia jasnosc (ogien)
    """
    zcr      = features["zcr"]
    sc       = features["sc"]
    flatness = features["flatness"]
    low_e    = features["low_energy"]
    kurt     = features["kurtosis"]

    scores = {cat: 0.0 for cat in CATEGORIES}

    if kurt > 8:
        # --- IMPULSOWE ---
        strength = min(kurt / 50.0, 1.0)
        if zcr > 0.12 and kurt > 15:
            scores["serie_bron_maszynowej"] += 2.5 + strength
        elif low_e > 0.55 and sc < 1800:
            scores["bomba_lotnicza"] += 2.0 + strength * 1.5
            scores["wystrzal_krab"]  += 1.5 + strength
        else:
            scores["wystrzal_krab"]  += 2.0 + strength * 1.5
            scores["bomba_lotnicza"] += 1.0 + strength
    else:
        # --- CIAGŁE ---
        if sc < 1200 and low_e > 0.45:
            scores["kolumna_pancerna"] += 2.5
            if flatness > 0.12:
                scores["kolumna_pancerna"] += 0.5

        if flatness > 0.2 and sc < 1500:
            scores["woda_wyciek"] += 2.0
            if zcr > 0.04:
                scores["woda_wyciek"] += 0.5

        if 1000 < sc < 3500 and zcr > 0.06 and flatness < 0.3:
            scores["pozar_trzask"] += 2.0
            if 0.08 < flatness < 0.25:
                scores["pozar_trzask"] += 0.5

        # wzmocnij najsilniejsza kategorie ciagla
        cont = {k: scores[k] for k in CONTINUOUS}
        if cont:
            best = max(cont, key=lambda k: cont[k])
            scores[best] += 0.3

    total    = sum(scores.values()) + 1e-9
    best_cat = max(scores, key=lambda k: scores[k])
    return best_cat, float(scores[best_cat] / total)


# ══════════════════════════════════════════════════════════════
# AKUMULATOR WYNIKOW
# ══════════════════════════════════════════════════════════════

class MissionResult:
    def __init__(self):
        self.score_sum  = defaultdict(float)
        self.vote_count = defaultdict(int)
        self.lock = threading.Lock()

    def update(self, category: str, confidence: float):
        with self.lock:
            self.score_sum[category]  += confidence
            self.vote_count[category] += 1

    def top3(self) -> list:
        with self.lock:
            ranked = sorted(self.score_sum.items(), key=lambda x: -x[1])
        result = []
        for cat, total in ranked[:3]:
            if total > 0:
                result.append({
                    "category":    cat,
                    "label":       LABELS_PL[cat],
                    "type":        "ciagly" if cat in CONTINUOUS else "przerywany",
                    "total_score": round(total, 2),
                    "detections":  self.vote_count[cat],
                })
        return result

    def summary(self) -> str:
        lines = [
            "\n" + "=" * 52,
            "  WYNIK MISJI -- TOP 3 WYKRYTE PROBKI",
            "=" * 52,
        ]
        for i, d in enumerate(self.top3(), 1):
            lines.append(
                f"  {i}. {d['label']}\n"
                f"     typ={d['type']}  score={d['total_score']:.1f}"
                f"  detekcji={d['detections']}"
            )
        lines.append("=" * 52)
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps({
            "timestamp":  datetime.now().isoformat(),
            "top3":       self.top3(),
            "all_scores": {k: round(v, 2) for k, v in self.score_sum.items()},
        }, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════
# TRYB: MIKROFON
# ══════════════════════════════════════════════════════════════

def run_microphone(duration: float, device=None):
    mission  = MissionResult()
    audio_q  = queue.Queue(maxsize=20)
    stop_evt = threading.Event()
    buffer   = np.zeros(WINDOW_SAMPLES, dtype=np.float32)

    def audio_callback(indata, frames, time_info, status):
        chunk = indata[:, 0].copy().astype(np.float32)
        try:
            audio_q.put_nowait(chunk)
        except queue.Full:
            pass

    def classifier_thread():
        nonlocal buffer
        while not stop_evt.is_set() or not audio_q.empty():
            try:
                chunk = audio_q.get(timeout=0.5)
            except queue.Empty:
                continue

            buffer = np.roll(buffer, -len(chunk))
            buffer[-len(chunk):] = chunk

            if np.sqrt(np.mean(buffer ** 2)) < SILENCE_RMS:
                continue

            feats      = extract_features(buffer)
            cat, conf  = classify(feats)

            if conf >= MIN_CONFIDENCE:
                mission.update(cat, conf)
                ts = datetime.now().strftime("%H:%M:%S")
                print(
                    f"[{ts}]  {LABELS_PL[cat]:<32}"
                    f"  pewnosc={conf:.2f}"
                    f"  kurt={feats['kurtosis']:.1f}"
                    f"  sc={feats['sc']:.0f} Hz"
                )

    print("=" * 52)
    print("  KLASYFIKATOR DZWIEKOW -- DSP ADVANCED")
    print(f"  Nasluch: {duration:.0f} s  |  SR={SAMPLE_RATE} Hz")
    print("  Ctrl+C aby przerwac wczesniej")
    print("=" * 52)

    t = threading.Thread(target=classifier_thread, daemon=True)
    t.start()

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=HOP_SAMPLES,
            device=device,
            callback=audio_callback,
        ):
            end = time.time() + duration
            while time.time() < end:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[INFO] Przerwano przez uzytkownika.")

    stop_evt.set()
    t.join(timeout=5)
    print(mission.summary())

    path = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w", encoding="utf-8") as f:
        f.write(mission.to_json())
    print(f"[INFO] Raport zapisany: {path}")


# ══════════════════════════════════════════════════════════════
# TRYB: PLIK WAV
# ══════════════════════════════════════════════════════════════

def run_file(path: str):
    print(f"[INFO] Ladowanie: {path}")
    audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    mission   = MissionResult()
    n_windows = (len(audio) - WINDOW_SAMPLES) // HOP_SAMPLES + 1
    print(f"[INFO] Dlugosc: {len(audio)/sr:.1f}s  |  okien: {n_windows}\n")

    for i in range(n_windows):
        start  = i * HOP_SAMPLES
        window = audio[start: start + WINDOW_SAMPLES]
        if len(window) < WINDOW_SAMPLES:
            break
        if np.sqrt(np.mean(window ** 2)) < SILENCE_RMS:
            continue

        feats     = extract_features(window)
        cat, conf = classify(feats)
        t_sec     = start / sr

        if conf >= MIN_CONFIDENCE:
            mission.update(cat, conf)
            print(
                f"[{t_sec:6.1f}s]  {LABELS_PL[cat]:<32}"
                f"  pewnosc={conf:.2f}"
                f"  kurt={feats['kurtosis']:.1f}"
            )

    print(mission.summary())

    path_out = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(path_out, "w", encoding="utf-8") as f:
        f.write(mission.to_json())
    print(f"[INFO] Raport zapisany: {path_out}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DSP Advanced -- klasyfikator dzwiekow")
    parser.add_argument("--file",         type=str,   help="sciezka do pliku WAV")
    parser.add_argument("--duration",     type=float, default=360.0,
                        help="czas nasluchu [s] (domyslnie 360)")
    parser.add_argument("--device",       type=int,   default=None,
                        help="numer urzadzenia audio")
    parser.add_argument("--list-devices", action="store_true",
                        help="wypisz urzadzenia audio i wyjdz")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
    elif args.file:
        run_file(args.file)
    else:
        run_microphone(args.duration, device=args.device)