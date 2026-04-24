"""
GUI Klasyfikatora Dźwięków — DSP Advanced + AI Powered
=======================================================
Wymagania: pip install librosa sounddevice numpy scipy onnxruntime
           (tkinter jest wbudowany w Pythona)

Uruchomienie:
    python polaczony_classifier_gui.py
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import queue
import time
import json
import os
import numpy as np
from collections import defaultdict
from datetime import datetime

import librosa
import sounddevice as sd
import onnxruntime as ort

# ══════════════════════════════════════════════════════════════
# PARAMETRY
# ══════════════════════════════════════════════════════════════
SAMPLE_RATE    = 16000
WINDOW_SEC     = 2.0
OVERLAP_SEC    = 1.0
SILENCE_RMS    = 0.008
MIN_CONFIDENCE = 0.25  # Obniżone dla lepszej czułości

WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)
HOP_SAMPLES    = int(SAMPLE_RATE * (WINDOW_SEC - OVERLAP_SEC))

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
    "woda_wyciek":           "Woda / wyciek",
    "pozar_trzask":          "Pożar / trzask",
    "wystrzal_krab":         "Wystrzał KRAB",
    "bomba_lotnicza":        "Bomba lotnicza",
    "kolumna_pancerna":      "Kolumna pancerna",
    "serie_bron_maszynowej": "Seria / broń maszynowa",
}
ICONS = {
    "woda_wyciek":           "~",
    "pozar_trzask":          "#",
    "wystrzal_krab":         "!",
    "bomba_lotnicza":        "*",
    "kolumna_pancerna":      "=",
    "serie_bron_maszynowej": ":",
}

# ══════════════════════════════════════════════════════════════
# KONFIGURACJA MODELU AI
# ══════════════════════════════════════════════════════════════
MODEL_PATH = "models/models/yamnet_classifier.onnx"

def load_ai_model():
    if os.path.exists(MODEL_PATH):
        try:
            return ort.InferenceSession(MODEL_PATH)
        except Exception as e:
            print(f"Błąd ładowania modelu ONNX: {e}")
    return None

session = load_ai_model()

# ══════════════════════════════════════════════════════════════
# LOGIKA AUDIO (AI Powered + Enhanced Features)
# ══════════════════════════════════════════════════════════════

def extract_enhanced_features(audio):
    """Rozszerzona ekstrakcja cech dla lepszej klasyfikacji"""
    rms      = float(np.sqrt(np.mean(audio ** 2)))
    zcr      = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
    sc       = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=audio)))
    stft     = np.abs(librosa.stft(audio))
    freqs    = librosa.fft_frequencies(sr=SAMPLE_RATE)
    low_mask = freqs < 500
    low_energy = float(stft[low_mask].sum() / (stft.sum() + 1e-9))
    mu       = np.mean(audio)
    sigma    = np.std(audio) + 1e-9
    kurtosis = float(np.mean(((audio - mu) / sigma) ** 4))

    # Dodatkowe cechy dla lepszej detekcji
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=SAMPLE_RATE)))
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=SAMPLE_RATE), axis=1)
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13), axis=1)

    return {
        "rms": rms, "zcr": zcr, "sc": sc,
        "flatness": flatness, "low_energy": low_energy, "kurtosis": kurtosis,
        "rolloff": rolloff, "chroma": chroma.tolist(), "mfcc": mfcc.tolist()
    }

def enhanced_classify_logic(features):
    """Ulepszona logika klasyfikacji oparta na cechach"""
    zcr  = features["zcr"]
    sc   = features["sc"]
    flat = features["flatness"]
    low  = features["low_energy"]
    kurt = features["kurtosis"]
    rolloff = features["rolloff"]

    scores = {cat: 0.0 for cat in CATEGORIES}

    # Logika oparta na kurtosis dla impulsowych dźwięków
    if kurt > 8:
        s = min(kurt / 50.0, 1.0)
        if zcr > 0.12 and kurt > 15:
            scores["serie_bron_maszynowej"] += 2.5 + s
        elif low > 0.55 and sc < 1800:
            scores["bomba_lotnicza"] += 2.0 + s * 1.5
            scores["wystrzal_krab"]  += 1.5 + s
        else:
            scores["wystrzal_krab"]  += 2.0 + s * 1.5
            scores["bomba_lotnicza"] += 1.0 + s
    else:
        # Dla dźwięków ciągłych
        if sc < 1200 and low > 0.45:
            scores["kolumna_pancerna"] += 2.5 + (0.5 if flat > 0.12 else 0)
        if flat > 0.2 and sc < 1500:
            scores["woda_wyciek"] += 2.0 + (0.5 if zcr > 0.04 else 0)
        if 1000 < sc < 3500 and zcr > 0.06 and flat < 0.3:
            scores["pozar_trzask"] += 2.0 + (0.5 if 0.08 < flat < 0.25 else 0)

        # Dodatkowe reguły dla lepszej detekcji
        if rolloff < 1000 and flat > 0.15:
            scores["woda_wyciek"] += 0.5
        if rolloff > 3000 and zcr > 0.08:
            scores["pozar_trzask"] += 0.5

        cont = {k: scores[k] for k in CONTINUOUS}
        if cont:
            scores[max(cont, key=lambda k: cont[k])] += 0.3

    total = sum(scores.values()) + 1e-9
    best  = max(scores, key=lambda k: scores[k])
    return best, float(scores[best] / total), scores

def classify(audio_buffer):
    """Główna funkcja klasyfikacji wykorzystująca AI + rozszerzoną analizę"""

    # 1. Obliczanie energii (odfiltrowanie ciszy)
    rms = float(np.sqrt(np.mean(audio_buffer ** 2)))
    if rms < SILENCE_RMS:
        return None, 0.0, {}, {}

    # 2. Ekstrakcja rozszerzonych cech
    feats = extract_enhanced_features(audio_buffer)
    all_scores = {cat: 0.0 for cat in CATEGORIES}

    # 3. Jeśli model AI jest dostępny, użyj go jako podstawy
    if session is not None:
        try:
            input_data = audio_buffer.astype(np.float32).reshape(1, -1)
            inputs = {session.get_inputs()[0].name: input_data}
            outputs = session.run(None, inputs)

            predictions = outputs[0][0]
            top_idx = np.argmax(predictions)
            detected_cat = CATEGORIES[top_idx % len(CATEGORIES)]
            ai_confidence = float(predictions[top_idx])

            all_scores[detected_cat] = ai_confidence

            # Jeśli AI ma wysoką pewność, zwróć wynik
            if ai_confidence >= MIN_CONFIDENCE:
                return detected_cat, ai_confidence, feats, all_scores

            # W przeciwnym razie, wzmocnij logiką cech
            logic_cat, logic_conf, logic_scores = enhanced_classify_logic(feats)

            # Połącz wyniki: jeśli logika ma wyższą pewność, użyj jej
            if logic_conf > ai_confidence:
                all_scores = {k: max(all_scores.get(k, 0), v) for k, v in logic_scores.items()}
                return logic_cat, logic_conf, feats, all_scores
            else:
                return detected_cat, ai_confidence, feats, all_scores

        except Exception as e:
            print(f"Błąd AI: {e}")
            # Przejdź do logiki cech

    # 4. Fallback do rozszerzonej logiki cech
    cat, conf, scores = enhanced_classify_logic(feats)
    all_scores = scores
    return cat, conf, feats, all_scores

# ══════════════════════════════════════════════════════════════
# KOLORY I STYL (military terminal)
# ══════════════════════════════════════════════════════════════
C = {
    "bg":        "#1a1f2e",
    "bg2":       "#222840",
    "bg3":       "#252b3d",
    "border":    "#3d5a8a",
    "green":     "#4dff91",
    "green_dim": "#27c96a",
    "green_dk":  "#0f5c32",
    "amber":     "#ffd166",
    "red":       "#ff6b6b",
    "cyan":      "#74d7f7",
    "white":     "#e8edf5",
    "gray":      "#8898b8",
    "bar_bg":    "#161b2c",
    "btn_start": "#1a6e3c",
    "btn_stop":  "#6e1a1a",
    "btn_ref":   "#2a3550",
    "btn_test":  "#1a4a6e",
    "panel_hdr": "#2d3a55",
}

FONT_MONO  = ("Courier New", 11)
FONT_MONO_S= ("Courier New", 10)
FONT_MONO_L= ("Courier New", 14, "bold")
FONT_HEAD  = ("Courier New", 12, "bold")
FONT_TINY  = ("Courier New", 9)

# ══════════════════════════════════════════════════════════════
# GŁÓWNA APLIKACJA
# ══════════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DSP ADVANCED // ACOUSTIC CLASSIFIER (AI + Enhanced)")
        self.configure(bg=C["bg"])
        self.resizable(False, False)

        # Stan
        self.running      = False
        self.stop_evt     = threading.Event()
        self.audio_q      = queue.Queue(maxsize=30)
        self.ui_q         = queue.Queue()
        self.score_sum    = defaultdict(float)
        self.vote_count   = defaultdict(int)
        self.start_time   = None
        self.duration_var = tk.IntVar(value=360)
        self.device_var   = tk.StringVar()
        self.devices      = []
        self.rms_history  = [0.0] * 40
        self.mic_test_running = False

        self._build_ui()
        self._refresh_devices()
        self._poll_ui_queue()
        self._animate_vu()

        # Log status AI
        if session is not None:
            self._log("MODUŁ AI (ONNX) ZAŁADOWANY POMYŚLNIE.", "info")
        else:
            self._log("BRAK MODELU AI — UŻYWAM ROZSZERZONEJ LOGIKI CECH.", "warning")

    def _build_ui(self):
        # Główny kontener
        main = tk.Frame(self, bg=C["bg"])
        main.pack(fill="both", expand=True, padx=10, pady=10)

        # Nagłówek
        hdr = tk.Label(main, text="DSP ADVANCED // ACOUSTIC CLASSIFIER", font=FONT_HEAD, bg=C["bg"], fg=C["cyan"])
        hdr.pack(pady=(0,10))

        # Panel kontrolny
        ctrl_frame = tk.Frame(main, bg=C["bg3"], relief="ridge", bd=2)
        ctrl_frame.pack(fill="x", pady=(0,10))

        # Przyciski
        btn_frame = tk.Frame(ctrl_frame, bg=C["bg3"])
        btn_frame.pack(pady=5)

        self.start_btn = tk.Button(btn_frame, text="START", command=self._start, bg=C["btn_start"], fg=C["white"], font=FONT_MONO, width=10)
        self.start_btn.pack(side="left", padx=5)

        self.stop_btn = tk.Button(btn_frame, text="STOP", command=self._stop, bg=C["btn_stop"], fg=C["white"], font=FONT_MONO, width=10, state="disabled")
        self.stop_btn.pack(side="left", padx=5)

        tk.Button(btn_frame, text="ODŚWIEŻ", command=self._refresh_devices, bg=C["btn_ref"], fg=C["white"], font=FONT_MONO, width=10).pack(side="left", padx=5)

        tk.Button(btn_frame, text="TEST MIC", command=self._test_mic, bg=C["btn_test"], fg=C["white"], font=FONT_MONO, width=10).pack(side="left", padx=5)

        # Selektor urządzenia
        dev_frame = tk.Frame(ctrl_frame, bg=C["bg3"])
        dev_frame.pack(pady=5)

        tk.Label(dev_frame, text="Urządzenie audio:", bg=C["bg3"], fg=C["white"], font=FONT_MONO).pack(side="left")
        self.dev_combo = ttk.Combobox(dev_frame, textvariable=self.device_var, state="readonly", width=40)
        self.dev_combo.pack(side="left", padx=5)

        # Czas trwania
        dur_frame = tk.Frame(ctrl_frame, bg=C["bg3"])
        dur_frame.pack(pady=5)

        tk.Label(dur_frame, text="Czas trwania (sek):", bg=C["bg3"], fg=C["white"], font=FONT_MONO).pack(side="left")
        tk.Spinbox(dur_frame, from_=10, to=3600, textvariable=self.duration_var, width=5, bg=C["bg2"], fg=C["white"], font=FONT_MONO).pack(side="left", padx=5)

        # VU Meter
        vu_frame = tk.Frame(main, bg=C["bg"], relief="ridge", bd=2)
        vu_frame.pack(fill="x", pady=(0,10))

        tk.Label(vu_frame, text="VU METER", font=FONT_HEAD, bg=C["bg"], fg=C["amber"]).pack(pady=(5,0))

        self.vu_canvas = tk.Canvas(vu_frame, height=60, bg=C["bar_bg"], highlightthickness=0)
        self.vu_canvas.pack(fill="x", padx=10, pady=5)

        # Główny panel wyników
        results_frame = tk.Frame(main, bg=C["bg"])
        results_frame.pack(fill="both", expand=True)

        # Panel klasyfikacji
        class_frame = tk.Frame(results_frame, bg=C["bg3"], relief="ridge", bd=2)
        class_frame.pack(side="left", fill="both", expand=True, padx=(0,5))

        tk.Label(class_frame, text="KLASYFIKACJA", font=FONT_HEAD, bg=C["panel_hdr"], fg=C["white"]).pack(fill="x")

        self.class_text = scrolledtext.ScrolledText(class_frame, height=15, bg=C["bg2"], fg=C["green"], font=FONT_MONO, wrap="word")
        self.class_text.pack(fill="both", expand=True, padx=5, pady=5)

        # Panel statystyk
        stats_frame = tk.Frame(results_frame, bg=C["bg3"], relief="ridge", bd=2)
        stats_frame.pack(side="right", fill="y", padx=(5,0))

        tk.Label(stats_frame, text="STATYSTYKI", font=FONT_HEAD, bg=C["panel_hdr"], fg=C["white"]).pack(fill="x")

        self.stats_text = tk.Text(stats_frame, height=15, bg=C["bg2"], fg=C["cyan"], font=FONT_MONO_S, wrap="word")
        self.stats_text.pack(fill="both", expand=True, padx=5, pady=5)

        # Panel logów
        log_frame = tk.Frame(main, bg=C["bg3"], relief="ridge", bd=2)
        log_frame.pack(fill="x", pady=(10,0))

        tk.Label(log_frame, text="LOG SYSTEMOWY", font=FONT_HEAD, bg=C["panel_hdr"], fg=C["white"]).pack(fill="x")

        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, bg=C["bg2"], fg=C["gray"], font=FONT_MONO_S, wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)

    def _refresh_devices(self):
        self.devices = sd.query_devices()
        dev_list = [f"{i}: {d['name']}" for i, d in enumerate(self.devices)]
        self.dev_combo['values'] = dev_list
        if dev_list:
            self.dev_combo.current(0)
        self._log("Urządzenia audio odświeżone.", "info")

    def _start(self):
        if self.running:
            return
        self.running = True
        self.start_time = time.time()
        self.stop_evt.clear()
        self.score_sum.clear()
        self.vote_count.clear()
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self._log("Rozpoczęto klasyfikację dźwięków.", "info")
        threading.Thread(target=self._audio_thread, daemon=True).start()

    def _stop(self):
        if not self.running:
            return
        self.running = False
        self.stop_evt.set()
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self._log("Zatrzymano klasyfikację.", "info")

    def _audio_thread(self):
        try:
            dev_idx = self.dev_combo.current()
            if dev_idx < 0:
                self.ui_q.put(("log", "Błąd: Nie wybrano urządzenia audio.", "error"))
                return

            duration = self.duration_var.get()
            end_time = time.time() + duration

            def callback(indata, frames, time_info, status):
                if status:
                    self.ui_q.put(("log", f"Status audio: {status}", "warning"))
                if not self.stop_evt.is_set():
                    try:
                        self.audio_q.put(indata[:, 0].copy(), timeout=0.1)
                    except queue.Full:
                        pass

            with sd.InputStream(device=dev_idx, channels=1, samplerate=SAMPLE_RATE, callback=callback):
                while time.time() < end_time and not self.stop_evt.is_set():
                    time.sleep(0.1)

        except Exception as e:
            self.ui_q.put(("log", f"Błąd audio: {e}", "error"))
        finally:
            self.running = False
            self.ui_q.put(("stop",))

    def _process_audio(self):
        buffer = []
        while not self.audio_q.empty():
            buffer.extend(self.audio_q.get())

        if len(buffer) < WINDOW_SAMPLES:
            return

        audio = np.array(buffer[-WINDOW_SAMPLES:], dtype=np.float32)

        cat, conf, feats, scores = classify(audio)

        if cat and conf >= MIN_CONFIDENCE:
            self.score_sum[cat] += conf
            self.vote_count[cat] += 1

            icon = ICONS.get(cat, "?")
            label = LABELS_PL.get(cat, cat)
            self.ui_q.put(("classify", f"{icon} {label} ({conf:.2f})", feats))

        # Aktualizacja VU
        rms = float(np.sqrt(np.mean(audio ** 2)))
        self.ui_q.put(("vu", rms))

    def _test_mic(self):
        if self.mic_test_running:
            return
        self.mic_test_running = True
        threading.Thread(target=self._mic_test_thread, daemon=True).start()

    def _mic_test_thread(self):
        try:
            dev_idx = self.dev_combo.current()
            if dev_idx < 0:
                self.ui_q.put(("log", "Błąd: Nie wybrano urządzenia audio.", "error"))
                return

            self._log("Test mikrofonu rozpoczęty (5 sek).", "info")

            buffer = []

            def callback(indata, frames, time_info, status):
                buffer.extend(indata[:, 0])

            with sd.InputStream(device=dev_idx, channels=1, samplerate=SAMPLE_RATE, callback=callback):
                time.sleep(5)

            if buffer:
                audio = np.array(buffer, dtype=np.float32)
                rms = float(np.sqrt(np.mean(audio ** 2)))
                self._log(f"Test mikrofonu zakończony. RMS: {rms:.4f}", "info")
            else:
                self._log("Test mikrofonu: Brak danych audio.", "warning")

        except Exception as e:
            self.ui_q.put(("log", f"Błąd testu mikrofonu: {e}", "error"))
        finally:
            self.mic_test_running = False

    def _poll_ui_queue(self):
        try:
            while True:
                msg = self.ui_q.get_nowait()
                if msg[0] == "classify":
                    self.class_text.insert("end", f"{datetime.now().strftime('%H:%M:%S')} {msg[1]}\n")
                    self.class_text.see("end")
                    self._update_stats()
                elif msg[0] == "log":
                    color = {"info": C["green"], "warning": C["amber"], "error": C["red"]}.get(msg[2], C["gray"])
                    self.log_text.insert("end", f"{datetime.now().strftime('%H:%M:%S')} {msg[1]}\n", ("color",))
                    self.log_text.tag_config("color", foreground=color)
                    self.log_text.see("end")
                elif msg[0] == "vu":
                    self.rms_history.append(msg[1])
                    self.rms_history.pop(0)
                elif msg[0] == "stop":
                    self._stop()
        except queue.Empty:
            pass
        self.after(100, self._poll_ui_queue)

    def _update_stats(self):
        self.stats_text.delete(1.0, "end")
        total_votes = sum(self.vote_count.values())
        if total_votes == 0:
            return

        sorted_cats = sorted(self.vote_count.keys(), key=lambda c: self.score_sum[c] / self.vote_count[c], reverse=True)

        for cat in sorted_cats:
            votes = self.vote_count[cat]
            avg_conf = self.score_sum[cat] / votes
            pct = votes / total_votes * 100
            icon = ICONS.get(cat, "?")
            label = LABELS_PL.get(cat, cat)
            self.stats_text.insert("end", f"{icon} {label}\n")
            self.stats_text.insert("end", f"   Głosy: {votes} ({pct:.1f}%)\n")
            self.stats_text.insert("end", f"   Średnia pewność: {avg_conf:.2f}\n\n")

    def _animate_vu(self):
        self.vu_canvas.delete("all")
        width = self.vu_canvas.winfo_width()
        height = self.vu_canvas.winfo_height()

        if width <= 1:
            self.after(100, self._animate_vu)
            return

        max_rms = max(self.rms_history) if self.rms_history else 0.01
        bar_width = width / len(self.rms_history)

        for i, rms in enumerate(self.rms_history):
            bar_height = (rms / max_rms) * height
            color = C["green"] if rms < 0.05 else C["amber"] if rms < 0.1 else C["red"]
            self.vu_canvas.create_rectangle(i * bar_width, height - bar_height, (i + 1) * bar_width, height, fill=color, outline="")

        self.after(100, self._animate_vu)

    def _log(self, msg, level="info"):
        self.ui_q.put(("log", msg, level))

    def _process_audio(self):
        buffer = []
        while not self.audio_q.empty():
            buffer.extend(self.audio_q.get())

        if len(buffer) < WINDOW_SAMPLES:
            return

        audio = np.array(buffer[-WINDOW_SAMPLES:], dtype=np.float32)

        cat, conf, feats, scores = classify(audio)

        if cat and conf >= MIN_CONFIDENCE:
            self.score_sum[cat] += conf
            self.vote_count[cat] += 1

            icon = ICONS.get(cat, "?")
            label = LABELS_PL.get(cat, cat)
            self.ui_q.put(("classify", f"{icon} {label} ({conf:.2f})", feats))

        # Aktualizacja VU
        rms = float(np.sqrt(np.mean(audio ** 2)))
        self.ui_q.put(("vu", rms))

    def _test_mic(self):
        if self.mic_test_running:
            return
        self.mic_test_running = True
        threading.Thread(target=self._mic_test_thread, daemon=True).start()

    def _mic_test_thread(self):
        try:
            dev_idx = self.dev_combo.current()
            if dev_idx < 0:
                self.ui_q.put(("log", "Błąd: Nie wybrano urządzenia audio.", "error"))
                return

            self._log("Test mikrofonu rozpoczęty (5 sek).", "info")

            buffer = []

            def callback(indata, frames, time_info, status):
                buffer.extend(indata[:, 0])

            with sd.InputStream(device=dev_idx, channels=1, samplerate=SAMPLE_RATE, callback=callback):
                time.sleep(5)

            if buffer:
                audio = np.array(buffer, dtype=np.float32)
                rms = float(np.sqrt(np.mean(audio ** 2)))
                self._log(f"Test mikrofonu zakończony. RMS: {rms:.4f}", "info")
            else:
                self._log("Test mikrofonu: Brak danych audio.", "warning")

        except Exception as e:
            self.ui_q.put(("log", f"Błąd testu mikrofonu: {e}", "error"))
        finally:
            self.mic_test_running = False

    def _poll_ui_queue(self):
        try:
            while True:
                msg = self.ui_q.get_nowait()
                if msg[0] == "classify":
                    self.class_text.insert("end", f"{datetime.now().strftime('%H:%M:%S')} {msg[1]}\n")
                    self.class_text.see("end")
                    self._update_stats()
                elif msg[0] == "log":
                    color = {"info": C["green"], "warning": C["amber"], "error": C["red"]}.get(msg[2], C["gray"])
                    self.log_text.insert("end", f"{datetime.now().strftime('%H:%M:%S')} {msg[1]}\n", ("color",))
                    self.log_text.tag_config("color", foreground=color)
                    self.log_text.see("end")
                elif msg[0] == "vu":
                    self.rms_history.append(msg[1])
                    self.rms_history.pop(0)
                elif msg[0] == "stop":
                    self._stop()
        except queue.Empty:
            pass
        self.after(100, self._poll_ui_queue)

    def _update_stats(self):
        self.stats_text.delete(1.0, "end")
        total_votes = sum(self.vote_count.values())
        if total_votes == 0:
            return

        sorted_cats = sorted(self.vote_count.keys(), key=lambda c: self.score_sum[c] / self.vote_count[c], reverse=True)

        for cat in sorted_cats:
            votes = self.vote_count[cat]
            avg_conf = self.score_sum[cat] / votes
            pct = votes / total_votes * 100
            icon = ICONS.get(cat, "?")
            label = LABELS_PL.get(cat, cat)
            self.stats_text.insert("end", f"{icon} {label}\n")
            self.stats_text.insert("end", f"   Głosy: {votes} ({pct:.1f}%)\n")
            self.stats_text.insert("end", f"   Średnia pewność: {avg_conf:.2f}\n\n")

    def _animate_vu(self):
        self.vu_canvas.delete("all")
        width = self.vu_canvas.winfo_width()
        height = self.vu_canvas.winfo_height()

        if width <= 1:
            self.after(100, self._animate_vu)
            return

        max_rms = max(self.rms_history) if self.rms_history else 0.01
        bar_width = width / len(self.rms_history)

        for i, rms in enumerate(self.rms_history):
            bar_height = (rms / max_rms) * height
            color = C["green"] if rms < 0.05 else C["amber"] if rms < 0.1 else C["red"]
            self.vu_canvas.create_rectangle(i * bar_width, height - bar_height, (i + 1) * bar_width, height, fill=color, outline="")

        self.after(100, self._animate_vu)


if __name__ == "__main__":
    app = App()
    app.mainloop()
