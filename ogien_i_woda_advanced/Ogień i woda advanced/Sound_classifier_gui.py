"""
GUI Klasyfikatora Dzwiekow — DSP Advanced
==========================================
Wymagania:  pip install librosa sounddevice numpy scipy
            (tkinter jest wbudowany w Pythona)

Uruchomienie:
    python sound_classifier_gui.py
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import queue
import time
import json
import numpy as np
from collections import defaultdict
from datetime import datetime

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
    "pozar_trzask":          "Pozar / trzask",
    "wystrzal_krab":         "Wystrzal KRAB",
    "bomba_lotnicza":        "Bomba lotnicza",
    "kolumna_pancerna":      "Kolumna pancerna",
    "serie_bron_maszynowej": "Serie / bron maszynowa",
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
# LOGIKA AUDIO (bez zmian wzgledem CLI)
# ══════════════════════════════════════════════════════════════

def extract_features(audio):
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
    return {"rms": rms, "zcr": zcr, "sc": sc,
            "flatness": flatness, "low_energy": low_energy, "kurtosis": kurtosis}

def classify(features):
    zcr  = features["zcr"]
    sc   = features["sc"]
    flat = features["flatness"]
    low  = features["low_energy"]
    kurt = features["kurtosis"]
    scores = {cat: 0.0 for cat in CATEGORIES}
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
        if sc < 1200 and low > 0.45:
            scores["kolumna_pancerna"] += 2.5 + (0.5 if flat > 0.12 else 0)
        if flat > 0.2 and sc < 1500:
            scores["woda_wyciek"] += 2.0 + (0.5 if zcr > 0.04 else 0)
        if 1000 < sc < 3500 and zcr > 0.06 and flat < 0.3:
            scores["pozar_trzask"] += 2.0 + (0.5 if 0.08 < flat < 0.25 else 0)
        cont = {k: scores[k] for k in CONTINUOUS}
        if cont:
            scores[max(cont, key=lambda k: cont[k])] += 0.3
    total = sum(scores.values()) + 1e-9
    best  = max(scores, key=lambda k: scores[k])
    return best, float(scores[best] / total), scores

# ══════════════════════════════════════════════════════════════
# KOLORY I STYL (military terminal)
# ══════════════════════════════════════════════════════════════
C = {
    "bg":        "#1a1f2e",   # granatowe tlo — mniej meczy oczy niz czarne
    "bg2":       "#222840",   # tlo przyciskow
    "bg3":       "#252b3d",   # tlo paneli
    "border":    "#3d5a8a",   # wyrazne niebieskie obramowanie
    "green":     "#4dff91",   # jasna mietowa zielen — bardzo czytelna
    "green_dim": "#27c96a",   # srednia zielen
    "green_dk":  "#0f5c32",   # ciemna zielen do barow
    "amber":     "#ffd166",   # ciepły zolty — czytelniejszy niz pomaranczowy
    "red":       "#ff6b6b",   # lagodniejsza czerwien
    "cyan":      "#74d7f7",   # jasny blekitny
    "white":     "#e8edf5",   # prawie bialy — bardzo czytelny
    "gray":      "#8898b8",   # sredni szary-niebieski
    "bar_bg":    "#161b2c",   # tlo barow
    "btn_start": "#1a6e3c",   # wypelnione tlo przycisku START
    "btn_stop":  "#6e1a1a",   # wypelnione tlo przycisku STOP
    "btn_ref":   "#2a3550",   # tlo przycisku ODSWIEZ
    "btn_test":  "#1a4a6e",   # tlo przycisku TEST
    "panel_hdr": "#2d3a55",   # tlo naglowkow paneli
}

FONT_MONO  = ("Courier New", 11)
FONT_MONO_S= ("Courier New", 10)
FONT_MONO_L= ("Courier New", 14, "bold")
FONT_HEAD  = ("Courier New", 12, "bold")
FONT_TINY  = ("Courier New", 9)

# ══════════════════════════════════════════════════════════════
# GLOWNA APLIKACJA
# ══════════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DSP ADVANCED // ACOUSTIC CLASSIFIER")
        self.configure(bg=C["bg"])
        self.resizable(False, False)

        # Stan
        self.running      = False
        self.stop_evt     = threading.Event()
        self.audio_q      = queue.Queue(maxsize=30)
        self.ui_q         = queue.Queue()          # wiadomosci do GUI
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

    # ──────────────────────────────────────────────────────────
    # BUDOWANIE UI
    # ──────────────────────────────────────────────────────────

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=C["bg"], pady=8)
        hdr.pack(fill="x", padx=16)
        tk.Label(hdr, text="[ DSP ADVANCED // ACOUSTIC SIGNAL CLASSIFIER v1.0 ]",
                 font=("Courier New", 13, "bold"), fg=C["amber"], bg=C["bg"]).pack()
        tk.Label(hdr, text="WOJSKOWY SYSTEM IDENTYFIKACJI DZWIEKOW BOJOWYCH",
                 font=("Courier New", 10), fg=C["gray"], bg=C["bg"]).pack()

        self._sep()

        # Row: mikrofon + czas
        row1 = tk.Frame(self, bg=C["bg"])
        row1.pack(fill="x", padx=16, pady=4)

        # -- Wybor mikrofonu --
        mic_box = self._panel(row1, ">> URZADZENIE AUDIO")
        mic_box.pack(side="left", fill="both", expand=True, padx=(0, 8))

        self.device_combo = ttk.Combobox(
            mic_box, textvariable=self.device_var,
            state="readonly", font=FONT_MONO_S, width=38,
        )
        self.device_combo.pack(fill="x", padx=6, pady=4)
        self._style_combobox()

        btn_row = tk.Frame(mic_box, bg=C["bg3"])
        btn_row.pack(fill="x", padx=6, pady=(0, 6))
        self._btn(btn_row, "ODSWIEZ", self._refresh_devices, C["gray"], bg_color=C["btn_ref"]).pack(side="left", padx=(0,6))
        self._btn(btn_row, "TEST MIKROFONU", self._toggle_mic_test, C["cyan"], bg_color=C["btn_test"]).pack(side="left")

        # VU meter
        vu_frame = tk.Frame(mic_box, bg=C["bg3"])
        vu_frame.pack(fill="x", padx=6, pady=(0, 8))
        tk.Label(vu_frame, text="POZIOM:", font=("Courier New", 9, "bold"), fg=C["white"], bg=C["bg3"]).pack(side="left")
        self.vu_canvas = tk.Canvas(vu_frame, height=14, bg=C["bar_bg"],
                                   highlightthickness=1, highlightbackground=C["border"])
        self.vu_canvas.pack(side="left", fill="x", expand=True, padx=(6,0))
        self.vu_label = tk.Label(vu_frame, text="--", font=FONT_TINY,
                                  fg=C["green"], bg=C["bg3"], width=5)
        self.vu_label.pack(side="left", padx=4)

        # -- Czas testu --
        time_box = self._panel(row1, ">> CZAS TESTU")
        time_box.pack(side="left", fill="both", padx=(0, 0))

        tf = tk.Frame(time_box, bg=C["bg3"])
        tf.pack(padx=6, pady=6)

        for label, val in [("30s", 30), ("60s", 60), ("110s", 110), ("360s", 360)]:
            tk.Radiobutton(
                tf, text=label, variable=self.duration_var, value=val,
                font=("Courier New", 10, "bold"),
                fg=C["white"], bg=C["btn_ref"],
                selectcolor=C["btn_start"],
                activebackground=C["btn_test"],
                activeforeground=C["white"],
                indicatoron=0,
                relief="raised", bd=2,
                padx=10, pady=6,
                cursor="hand2",
            ).pack(side="left", padx=3)

        tk.Label(time_box, text="lub wpisz (sekundy):", font=("Courier New", 9, "bold"), fg=C["white"], bg=C["bg3"]).pack(pady=(6,0))
        self.custom_time = tk.Entry(
            time_box, font=("Courier New", 12, "bold"), width=8,
            bg=C["bg2"], fg=C["amber"], insertbackground=C["amber"],
            relief="sunken", bd=3, justify="center",
        )
        self.custom_time.pack(pady=(2, 8))
        self.custom_time.insert(0, "")

        self._sep()

        # Pasek postępu + zegar
        prog_frame = tk.Frame(self, bg=C["bg"], padx=16)
        prog_frame.pack(fill="x", pady=(4,0))

        self.time_label = tk.Label(
            prog_frame, text="CZAS: 00:00 / 00:00",
            font=FONT_HEAD, fg=C["amber"], bg=C["bg"]
        )
        self.time_label.pack(side="left")

        self.status_dot = tk.Label(prog_frame, text="  ●  STAND-BY",
                                    font=FONT_HEAD, fg=C["gray"], bg=C["bg"])
        self.status_dot.pack(side="left", padx=16)

        self.progress = tk.Canvas(
            self, height=8, bg=C["bar_bg"],
            highlightthickness=1, highlightbackground=C["border"]
        )
        self.progress.pack(fill="x", padx=16, pady=6)
        self.prog_bar = self.progress.create_rectangle(0, 0, 0, 8, fill=C["green"], outline="")

        # Przyciski START/STOP
        btn_frame = tk.Frame(self, bg=C["bg"])
        btn_frame.pack(pady=8)
        self.btn_start = self._btn(btn_frame, "  ▶  START  ", self._start, C["green"], large=True, bg_color=C["btn_start"])
        self.btn_start.pack(side="left", padx=10)
        self.btn_stop = self._btn(btn_frame, "  ■  STOP  ", self._stop, C["red"], large=True, bg_color=C["btn_stop"])
        self.btn_stop.pack(side="left", padx=10)
        self.btn_stop.config(state="disabled")

        self._sep()

        # Log + wyniki side by side
        bottom = tk.Frame(self, bg=C["bg"])
        bottom.pack(fill="both", expand=True, padx=16, pady=(4, 12))

        # Log
        log_panel = self._panel(bottom, ">> LOG DETEKCJI")
        log_panel.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self.log = scrolledtext.ScrolledText(
            log_panel, width=52, height=14,
            font=("Courier New", 10), bg=C["bg"], fg=C["green"],
            insertbackground=C["green"], relief="flat",
            bd=6, wrap="word", state="disabled",
        )
        self.log.pack(fill="both", expand=True, padx=4, pady=4)
        self.log.tag_config("ts",    foreground=C["gray"])
        self.log.tag_config("cat",   foreground=C["green"])
        self.log.tag_config("warn",  foreground=C["amber"])
        self.log.tag_config("info",  foreground=C["cyan"])
        self.log.tag_config("err",   foreground=C["red"])
        self.log.tag_config("imp",   foreground=C["green"], font=("Courier New", 9, "bold"))

        # Panel wynikow
        res_panel = self._panel(bottom, ">> WYNIKI MISJI")
        res_panel.pack(side="left", fill="y", padx=(0, 0))

        self.result_frames = {}
        self.result_bars   = {}
        self.result_labels = {}
        self.result_scores = {}

        for i, cat in enumerate(CATEGORIES):
            f = tk.Frame(res_panel, bg=C["bg3"], pady=3)
            f.pack(fill="x", padx=6, pady=2)

            icon = tk.Label(f, text=ICONS[cat], font=("Courier New", 14, "bold"),
                            fg=C["green_dim"], bg=C["bg3"], width=2)
            icon.pack(side="left")

            info = tk.Frame(f, bg=C["bg3"])
            info.pack(side="left", fill="x", expand=True)

            name_row = tk.Frame(info, bg=C["bg3"])
            name_row.pack(fill="x")
            lbl = tk.Label(name_row, text=LABELS_PL[cat], font=("Courier New", 10, "bold"),
                           fg=C["white"], bg=C["bg3"], anchor="w")
            lbl.pack(side="left")
            typ = "CIAGLY" if cat in CONTINUOUS else "PRZERYWANY"
            tk.Label(name_row, text=f"[{typ}]", font=("Courier New", 9),
                     fg=C["cyan"], bg=C["bg3"]).pack(side="right")

            bar_bg = tk.Canvas(info, height=10, bg=C["bar_bg"],
                               highlightthickness=0)
            bar_bg.pack(fill="x", pady=1)
            bar = bar_bg.create_rectangle(0, 0, 0, 10, fill=C["green_dk"], outline="")

            score_lbl = tk.Label(info, text="0 detekcji  |  score: 0.0",
                                 font=("Courier New", 9), fg=C["gray"], bg=C["bg3"], anchor="w")
            score_lbl.pack(fill="x")

            self.result_frames[cat] = f
            self.result_bars[cat]   = (bar_bg, bar)
            self.result_scores[cat] = score_lbl

        # Wiersz TOP wynikow
        self.top_label = tk.Label(
            res_panel,
            text="--- brak danych ---",
            font=("Courier New", 10, "bold"),
            fg=C["amber"], bg=C["panel_hdr"],
            justify="left", anchor="w",
            wraplength=260,
            padx=8, pady=6,
        )
        self.top_label.pack(fill="x", padx=6, pady=(6, 4))

    # ──────────────────────────────────────────────────────────
    # HELPERY UI
    # ──────────────────────────────────────────────────────────

    def _sep(self):
        tk.Frame(self, height=1, bg=C["border"]).pack(fill="x", padx=16, pady=4)

    def _panel(self, parent, title):
        outer = tk.Frame(parent, bg=C["bg3"],
                         highlightthickness=1, highlightbackground=C["border"])
        hdr_bar = tk.Frame(outer, bg=C["panel_hdr"])
        hdr_bar.pack(fill="x")
        tk.Label(hdr_bar, text=title, font=("Courier New", 10, "bold"),
                 fg=C["cyan"], bg=C["panel_hdr"], anchor="w",
                 padx=8, pady=4).pack(fill="x")
        tk.Frame(outer, height=1, bg=C["border"]).pack(fill="x")
        return outer

    def _btn(self, parent, text, cmd, color, large=False, bg_color=None):
        font = ("Courier New", 12, "bold") if large else ("Courier New", 10, "bold")
        bg   = bg_color if bg_color else C["bg2"]
        b = tk.Button(
            parent, text=text, command=cmd, font=font,
            fg=C["white"], bg=bg,
            activeforeground=C["bg"],
            activebackground=color,
            relief="raised", bd=2,
            padx=18 if large else 12,
            pady=10 if large else 6,
            cursor="hand2",
        )
        return b

    def _style_combobox(self):
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TCombobox",
            fieldbackground=C["bg2"],
            background=C["btn_ref"],
            foreground=C["white"],
            arrowcolor=C["cyan"],
            bordercolor=C["border"],
            lightcolor=C["border"],
            darkcolor=C["border"],
            selectbackground=C["btn_test"],
            selectforeground=C["white"],
            padding=4,
        )

    def _log(self, msg, tag="cat"):
        self.log.config(state="normal")
        ts = datetime.now().strftime("%H:%M:%S")
        self.log.insert("end", f"[{ts}] ", "ts")
        self.log.insert("end", msg + "\n", tag)
        self.log.see("end")
        self.log.config(state="disabled")

    # ──────────────────────────────────────────────────────────
    # URZADZENIA AUDIO
    # ──────────────────────────────────────────────────────────

    def _refresh_devices(self):
        self.devices = []
        names = []
        try:
            devs = sd.query_devices()
            for i, d in enumerate(devs):
                if d["max_input_channels"] > 0:
                    name = f"[{i}] {d['name'][:40]}"
                    self.devices.append(i)
                    names.append(name)
        except Exception as e:
            self._log(f"Blad: {e}", "err")

        self.device_combo["values"] = names
        if names:
            self.device_combo.current(0)
            self._log(f"Znaleziono {len(names)} urzadzen wejsciowych.", "info")
        else:
            self._log("Brak urzadzen audio!", "err")

    def _get_device_index(self):
        idx = self.device_combo.current()
        if idx >= 0 and idx < len(self.devices):
            return self.devices[idx]
        return None

    # ──────────────────────────────────────────────────────────
    # TEST MIKROFONU
    # ──────────────────────────────────────────────────────────

    def _toggle_mic_test(self):
        if not self.mic_test_running:
            self.mic_test_running = True
            self._log("Test mikrofonu uruchomiony...", "info")
            threading.Thread(target=self._mic_test_thread, daemon=True).start()
        else:
            self.mic_test_running = False
            self._log("Test mikrofonu zatrzymany.", "info")

    def _mic_test_thread(self):
        dev = self._get_device_index()
        test_q = queue.Queue()

        def cb(indata, frames, t, status):
            test_q.put(indata[:, 0].copy())

        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                dtype="float32", blocksize=HOP_SAMPLES,
                                device=dev, callback=cb):
                while self.mic_test_running:
                    try:
                        chunk = test_q.get(timeout=0.3)
                        rms   = float(np.sqrt(np.mean(chunk ** 2)))
                        self.ui_q.put(("vu", rms))
                    except queue.Empty:
                        pass
        except Exception as e:
            self.ui_q.put(("log_err", f"Blad testu: {e}"))
            self.mic_test_running = False

        self.ui_q.put(("vu", 0.0))

    # ──────────────────────────────────────────────────────────
    # VU METER ANIMACJA
    # ──────────────────────────────────────────────────────────

    def _animate_vu(self):
        rms = self.rms_history[-1]
        w   = self.vu_canvas.winfo_width()
        if w > 1:
            level = min(rms / 0.15, 1.0)
            fill_w = int(w * level)
            color  = C["green"] if level < 0.6 else (C["amber"] if level < 0.85 else C["red"])
            self.vu_canvas.delete("all")
            self.vu_canvas.create_rectangle(0, 0, fill_w, 14, fill=color, outline="")
            db = 20 * np.log10(rms + 1e-9)
            self.vu_label.config(text=f"{db:.0f}dB", fg=color)
        self.after(80, self._animate_vu)

    # ──────────────────────────────────────────────────────────
    # PASEK POSTEPU
    # ──────────────────────────────────────────────────────────

    def _update_progress(self):
        if not self.running:
            return
        elapsed  = time.time() - self.start_time
        duration = self._get_duration()
        frac     = min(elapsed / duration, 1.0)
        w        = self.progress.winfo_width()
        self.progress.coords(self.prog_bar, 0, 0, int(w * frac), 8)
        e_str = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"
        d_str = f"{int(duration//60):02d}:{int(duration%60):02d}"
        self.time_label.config(text=f"CZAS: {e_str} / {d_str}")
        if elapsed < duration:
            self.after(500, self._update_progress)
        else:
            self._stop()

    def _get_duration(self):
        custom = self.custom_time.get().strip()
        if custom.isdigit():
            return float(custom)
        return float(self.duration_var.get())

    # ──────────────────────────────────────────────────────────
    # START / STOP
    # ──────────────────────────────────────────────────────────

    def _start(self):
        if self.running:
            return
        # reset
        self.score_sum    = defaultdict(float)
        self.vote_count   = defaultdict(int)
        self.stop_evt     = threading.Event()
        self.audio_q      = queue.Queue(maxsize=30)
        self.running      = True
        self.start_time   = time.time()

        if self.mic_test_running:
            self.mic_test_running = False

        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.status_dot.config(text="  ●  NASLUCH AKTYWNY", fg=C["green"])
        self.progress.coords(self.prog_bar, 0, 0, 0, 8)

        self._log("=" * 44, "info")
        self._log("MISJA ROZPOCZETA", "imp")
        self._log(f"Czas: {self._get_duration():.0f}s  |  SR={SAMPLE_RATE}Hz", "info")
        self._log("=" * 44, "info")

        threading.Thread(target=self._mic_thread, daemon=True).start()
        threading.Thread(target=self._classifier_thread, daemon=True).start()
        self._update_progress()

    def _stop(self):
        if not self.running:
            return
        self.running = False
        self.stop_evt.set()
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.status_dot.config(text="  ●  STAND-BY", fg=C["gray"])
        self._log("MISJA ZATRZYMANA.", "warn")
        self._show_final()

    # ──────────────────────────────────────────────────────────
    # WATKI AUDIO
    # ──────────────────────────────────────────────────────────

    def _mic_thread(self):
        dev = self._get_device_index()

        def cb(indata, frames, t, status):
            chunk = indata[:, 0].copy().astype(np.float32)
            rms   = float(np.sqrt(np.mean(chunk ** 2)))
            self.ui_q.put(("vu", rms))
            try:
                self.audio_q.put_nowait(chunk)
            except queue.Full:
                pass

        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                dtype="float32", blocksize=HOP_SAMPLES,
                                device=dev, callback=cb):
                while not self.stop_evt.is_set():
                    time.sleep(0.05)
        except Exception as e:
            self.ui_q.put(("log_err", f"Blad mikrofonu: {e}"))
            self.stop_evt.set()

        self.ui_q.put(("vu", 0.0))

    def _classifier_thread(self):
        buffer = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        while not self.stop_evt.is_set() or not self.audio_q.empty():
            try:
                chunk = self.audio_q.get(timeout=0.5)
            except queue.Empty:
                continue

            buffer = np.roll(buffer, -len(chunk))
            buffer[-len(chunk):] = chunk

            rms = float(np.sqrt(np.mean(buffer ** 2)))
            if rms < SILENCE_RMS:
                continue

            feats          = extract_features(buffer)
            cat, conf, all_scores = classify(feats)

            if conf >= MIN_CONFIDENCE:
                self.score_sum[cat]  += conf
                self.vote_count[cat] += 1
                self.ui_q.put(("detection", cat, conf, feats, all_scores))

    # ──────────────────────────────────────────────────────────
    # POLLING KOLEJKI UI
    # ──────────────────────────────────────────────────────────

    def _poll_ui_queue(self):
        try:
            while True:
                msg = self.ui_q.get_nowait()
                if msg[0] == "vu":
                    self.rms_history.append(msg[1])
                    self.rms_history = self.rms_history[-40:]
                elif msg[0] == "detection":
                    _, cat, conf, feats, all_scores = msg
                    self._on_detection(cat, conf, feats, all_scores)
                elif msg[0] == "log_err":
                    self._log(msg[1], "err")
        except queue.Empty:
            pass
        self.after(100, self._poll_ui_queue)

    def _on_detection(self, cat, conf, feats, all_scores):
        label = LABELS_PL[cat]
        icon  = ICONS[cat]
        self._log(
            f"{icon} {label:<24} [{conf:.2f}]  kurt={feats['kurtosis']:.1f}",
            "cat" if conf > 0.6 else "warn"
        )
        self._update_score_bars()

    def _update_score_bars(self):
        max_score = max(self.score_sum.values()) if self.score_sum else 1.0
        top3 = sorted(self.score_sum.items(), key=lambda x: -x[1])[:3]
        top3_cats = [t[0] for t in top3]

        for cat in CATEGORIES:
            score = self.score_sum.get(cat, 0.0)
            votes = self.vote_count.get(cat, 0)
            bar_cv, bar_id = self.result_bars[cat]
            w = bar_cv.winfo_width()
            frac = score / (max_score + 1e-9)
            fill_w = int(w * frac)
            is_top = cat in top3_cats[:3]
            color = C["green"] if cat == top3_cats[0] else (C["green_dim"] if is_top else C["green_dk"])
            bar_cv.coords(bar_id, 0, 0, fill_w, 10)
            bar_cv.itemconfig(bar_id, fill=color)
            self.result_scores[cat].config(
                text=f"{votes} detekcji  |  score: {score:.1f}",
                fg=C["green"] if is_top else C["gray"]
            )

        # TOP label
        lines = []
        for i, (c, s) in enumerate(top3, 1):
            typ = "CIG" if c in CONTINUOUS else "PRZ"
            lines.append(f"  {i}. [{typ}] {LABELS_PL[c]}  ({s:.1f})")
        self.top_label.config(text="\n".join(lines) if lines else "--- brak danych ---")

    def _show_final(self):
        top3 = sorted(self.score_sum.items(), key=lambda x: -x[1])[:3]
        self._log("=" * 44, "info")
        self._log("WYNIK MISJI — TOP 3:", "imp")
        for i, (cat, score) in enumerate(top3, 1):
            typ = "CIAGLY" if cat in CONTINUOUS else "PRZERYWANY"
            self._log(
                f"  {i}. {LABELS_PL[cat]} [{typ}]  score={score:.1f}  n={self.vote_count[cat]}",
                "imp"
            )
        self._log("=" * 44, "info")

        # Zapis JSON
        report = {
            "timestamp": datetime.now().isoformat(),
            "top3": [
                {"rank": i+1, "category": c, "label": LABELS_PL[c],
                 "type": "ciagly" if c in CONTINUOUS else "przerywany",
                 "score": round(s, 2), "detections": self.vote_count[c]}
                for i, (c, s) in enumerate(top3)
            ],
            "all_scores": {k: round(v, 2) for k, v in self.score_sum.items()},
        }
        path = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        self._log(f"Raport zapisany: {path}", "info")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = App()
    app.mainloop()
