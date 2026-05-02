"""
MIERNIK DECYBELI + KLASYFIKATOR DZWIEKOW — Droniada 2026
=========================================================
Wymagania:
    pip install sounddevice numpy librosa scikit-learn

Uruchomienie:
    python db_meter.py

Model: model_gotowy.pkl (musi byc w tym samym folderze co skrypt)
"""

import tkinter as tk
from tkinter import ttk
import threading
import queue
import time
import math
import pickle
import os
import numpy as np

try:
    import sounddevice as sd
    SD_OK = True
except ImportError:
    SD_OK = False

try:
    import librosa
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False

# ─────────────────────────────────────────────────────────────
# STALE
# ─────────────────────────────────────────────────────────────
SAMPLE_RATE   = 48000
BLOCK_SIZE    = 4096
HISTORY_LEN   = 200
DB_MIN        = -80.0
DB_MAX        = 0.0
REFRESH_MS    = 60

ZONE_YELLOW   = -6.0
ZONE_RED      = -3.0

# Ile sekund audio zbieramy przed klasyfikacja
# 2 sekundy = wystarczy dla MFCC, nie za dlugo ale zeby cos robilo
#test xd
KLASYFIKUJ_CO_SEK = 1.0

# Sciezka do modelu — szuka w tym samym folderze co skrypt
MODEL_SCIEZKA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_gotowy.pkl")

# Kolory
BG     = "#0d1117"
BG2    = "#161b22"
BG3    = "#21262d"
ACCENT = "#58a6ff"
GREEN  = "#3fb950"
YELLOW = "#d29922"
RED    = "#f85149"
TEXT   = "#e6edf3"
MUTED  = "#8b949e"
BORDER = "#30363d"

# Kolory dla kazdej klasy dzwieku
KOLORY_KLAS = {
    "bomba_lotnicza":         "#f85149",   # czerwony
    "kolumna_pancerna":       "#d29922",   # zolty
    "pozar_trzask":           "#ff7b00",   # pomaranczowy
    "serie_broni_maszynowej": "#a371f7",   # fioletowy
    "woda_wyciek":            "#58a6ff",   # niebieski
    "wystrzal_krab":          "#f0883e",   # lososiowy
}

NAZWY_PL = {
    "bomba_lotnicza":         "Bomba lotnicza",
    "kolumna_pancerna":       "Kolumna pancerna",
    "pozar_trzask":           "Pozar (trzask)",
    "serie_broni_maszynowej": "Serie z broni maszynowej",
    "woda_wyciek":            "Woda (wyciek)",
    "wystrzal_krab":          "Wystrzal Krab",
}


# ─────────────────────────────────────────────────────────────
# LADOWANIE MODELU
# ─────────────────────────────────────────────────────────────

def zaladuj_model(sciezka):
    """
    Wczytuje model SVM + scaler z pliku .pkl.
    Zwraca slownik z modelem albo None jesli blad.
    """
    if not os.path.exists(sciezka):
        return None
    try:
        with open(sciezka, "rb") as f:
            pakiet = pickle.load(f)
        return pakiet
    except Exception as e:
        print(f"Blad ladowania modelu: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# EKSTRAKCJA CECH (identyczna jak w trenuj_model.py!)
# ─────────────────────────────────────────────────────────────

def wyciagnij_cechy(audio, sr):
    """
    WAZNE: Ta funkcja musi byc IDENTYCZNA jak w trenuj_model.py.
    Jesli cechy beda inne niz podczas treningu — model da zle wyniki.
    """
    cechy = []

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    cechy.extend(np.mean(mfcc, axis=1))
    cechy.extend(np.std(mfcc,  axis=1))
    delta = librosa.feature.delta(mfcc)
    cechy.extend(np.mean(delta, axis=1))

    sc = librosa.feature.spectral_centroid(y=audio, sr=sr)
    cechy.append(float(np.mean(sc)))
    cechy.append(float(np.std(sc)))

    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)
    cechy.append(float(np.mean(rolloff)))
    cechy.append(float(np.std(rolloff)))

    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    cechy.append(float(np.mean(bandwidth)))
    cechy.append(float(np.std(bandwidth)))

    zcr = librosa.feature.zero_crossing_rate(audio)
    cechy.append(float(np.mean(zcr)))
    cechy.append(float(np.std(zcr)))

    rms_feat = librosa.feature.rms(y=audio)
    cechy.append(float(np.mean(rms_feat)))
    cechy.append(float(np.std(rms_feat)))

    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    cechy.extend(np.mean(contrast, axis=1))

    stft     = np.abs(librosa.stft(audio))
    freqs    = librosa.fft_frequencies(sr=sr)
    low_e    = float(stft[freqs < 500].sum() / (stft.sum() + 1e-9))
    cechy.append(low_e)

    mu   = np.mean(audio)
    sig  = np.std(audio) + 1e-9
    kurt = float(np.mean(((audio - mu) / sig) ** 4))
    cechy.append(min(kurt, 100.0))

    return np.array(cechy, dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# WATEK KLASYFIKACJI (osobny od audio zeby nie spowalnic)
# ─────────────────────────────────────────────────────────────

class KlasyfikatorWatek:
    """
    Osobny watek ktory odbiera audio z kolejki,
    analizuje i odsyla wynik klasyfikacji do GUI.

    Dlaczego osobny watek?
    Ekstrakcja MFCC trwa ~50-200ms — za dlugo dla callbacku audio.
    Osobny watek przetwarza w tle nie blokujac nagrywania.
    """
    def __init__(self, model_pakiet, wynik_queue):
        self.model   = model_pakiet["svm"]
        self.scaler  = model_pakiet["scaler"]
        self.klasy   = model_pakiet["klasy"]
        self.sr_model = model_pakiet.get("sr", 22050)

        self.wynik_q  = wynik_queue       # tu wysylamy wynik do GUI
        self.audio_q  = queue.Queue(maxsize=5)  # tu dostajemy audio
        self.running  = False
        self._thread  = None

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._petla, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def dodaj_audio(self, audio_chunk):
        """Dodaje chunk audio do kolejki do analizy."""
        try:
            self.audio_q.put_nowait(audio_chunk)
        except queue.Full:
            pass  # jesli nie nadazamy — pomijamy, nie crashujemy

    def _petla(self):
        bufor = []
        # Ile probek potrzebujemy do klasyfikacji (2 sekundy w SR modelu)
        potrzebne = int(self.sr_model * KLASYFIKUJ_CO_SEK)

        while self.running:
            try:
                chunk = self.audio_q.get(timeout=0.5)
                bufor.append(chunk)

                # Sprawdz czy zebralismy wystarczajaco duzo audio
                lacznie = sum(len(c) for c in bufor)

                if lacznie >= potrzebne:
                    # Sklej wszystkie chunki w jeden array
                    audio = np.concatenate(bufor)

                    # Przeprobkuj z SR karty dzwiekowej do SR modelu
                    # (model trenowany na 22050 Hz, karta moze miec 48000 Hz)
                    if SAMPLE_RATE != self.sr_model:
                        audio = librosa.resample(
                            audio,
                            orig_sr=SAMPLE_RATE,
                            target_sr=self.sr_model
                        )

                    # Wez srodkowe 2 sekundy (odetnij ewentualne smieci na krawedzi)
                    okno = int(self.sr_model * KLASYFIKUJ_CO_SEK)
                    if len(audio) > okno:
                        start = (len(audio) - okno) // 2
                        audio = audio[start : start + okno]

                    # Sprawdz czy to nie cisza
                    rms = float(np.sqrt(np.mean(audio ** 2)))
                    if rms < 0.0005:
                        bufor = []
                        try:
                            self.wynik_q.put_nowait({
                                "klasa":       "—",
                                "pewnosc":     0.0,
                                "wszystkie":   {},
                                "cisza":       True,
                            })
                        except queue.Full:
                            pass
                        continue

                    # Wyciagnij cechy
                    cechy = wyciagnij_cechy(audio, self.sr_model)

                    # Skaluj (tak samo jak podczas treningu)
                    cechy_scaled = self.scaler.transform(cechy.reshape(1, -1))

                    # Klasyfikuj — SVM zwraca prawdopodobienstwa dla kazdej klasy
                    probs = self.model.predict_proba(cechy_scaled)[0]

                    # Znajdz klase z najwyzszym prawdopodobienstwem
                    best_idx  = int(np.argmax(probs))
                    best_klas = self.klasy[best_idx]
                    best_conf = float(probs[best_idx])

                    # Przygotuj wyniki dla wszystkich klas (do wyswietlenia w GUI)
                    wszystkie = {
                        self.klasy[i]: float(probs[i])
                        for i in range(len(self.klasy))
                    }

                    # Wyslij wynik do GUI
                    try:
                        self.wynik_q.put_nowait({
                            "klasa":     best_klas,
                            "pewnosc":   best_conf,
                            "wszystkie": wszystkie,
                            "cisza":     False,
                        })
                    except queue.Full:
                        pass

                    # Wyczysc bufor — zacznij zbierac od nowa
                    bufor = []

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Blad klasyfikacji: {e}")
                bufor = []


# ─────────────────────────────────────────────────────────────
# WATEK AUDIO
# ─────────────────────────────────────────────────────────────

class AudioWatek:
    def __init__(self, gui_queue, klasyfikator=None):
        self.gui_q        = gui_queue
        self.klasyfikator = klasyfikator
        self.running      = False
        self.thread       = None
        self._peak        = DB_MIN
        self._peak_time   = 0.0

    def start(self, device_idx, channels):
        self.stop()
        self.running    = True
        self._peak      = DB_MIN
        self._peak_time = time.time()
        self.thread = threading.Thread(
            target=self._petla,
            args=(device_idx, channels),
            daemon=True
        )
        self.thread.start()

    def _petla(self, device_idx, channels):
        def callback(indata, frames, time_info, status):
            # ── Oblicz dB ──────────────────────────────────────
            ch_dbs = []
            for c in range(indata.shape[1]):
                rms = float(np.sqrt(np.mean(indata[:, c] ** 2)))
                ch_dbs.append(20.0 * math.log10(max(rms, 1e-10)))

            avg_db = sum(ch_dbs) / len(ch_dbs)
            avg_db = max(DB_MIN, min(DB_MAX, avg_db))

            now = time.time()
            if avg_db >= self._peak:
                self._peak      = avg_db
                self._peak_time = now
            elif now - self._peak_time > 2.0:
                self._peak = max(avg_db, self._peak - 1.5)

            # ── Wyslij dB do GUI ───────────────────────────────
            try:
                self.gui_q.put_nowait({
                    "db":       avg_db,
                    "peak":     self._peak,
                    "channels": ch_dbs,
                })
            except queue.Full:
                pass

            # ── Wyslij audio do klasyfikatora ──────────────────
            # Bierzemy mono (srednia kanalow) bo model trenowany na mono
            if self.klasyfikator is not None:
                mono = indata.mean(axis=1).copy()
                self.klasyfikator.dodaj_audio(mono)

        try:
            with sd.InputStream(
                samplerate = SAMPLE_RATE,
                blocksize  = BLOCK_SIZE,
                device     = device_idx,
                channels   = channels,
                dtype      = "float32",
                callback   = callback,
            ):
                while self.running:
                    time.sleep(0.05)
        except Exception as e:
            try:
                self.gui_q.put_nowait({"error": str(e)})
            except queue.Full:
                pass
        self.running = False

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.5)
            self.thread = None


# ─────────────────────────────────────────────────────────────
# VU-METR
# ─────────────────────────────────────────────────────────────

class VUMeter(tk.Canvas):
    def __init__(self, parent, label="L", **kw):
        super().__init__(parent, bg=BG2, highlightthickness=0,
                         width=36, height=260, **kw)
        self.label = label
        self._db   = DB_MIN
        self._peak = DB_MIN
        self.bind("<Configure>", lambda e: self._draw())

    def set(self, db, peak):
        self._db   = db
        self._peak = peak
        self._draw()

    def _draw(self):
        self.delete("all")
        w = self.winfo_width(); h = self.winfo_height()
        if w < 4 or h < 4:
            return
        pt = 8; pb = 20; bh = h - pt - pb
        self.create_rectangle(2, pt, w-2, pt+bh, fill=BG3, outline=BORDER)

        frac   = max(0.0, (self._db - DB_MIN) / (DB_MAX - DB_MIN))
        fill_h = int(frac * bh)
        if fill_h > 0:
            r_h = int((DB_MAX - ZONE_RED)    / (DB_MAX - DB_MIN) * bh)
            y_h = int((ZONE_RED - ZONE_YELLOW)/ (DB_MAX - DB_MIN) * bh)
            g_h = bh - r_h - y_h
            yb  = pt + bh; rem = fill_h
            seg = min(rem, g_h)
            if seg > 0:
                self.create_rectangle(3, yb-seg, w-3, yb, fill=GREEN, outline="")
                yb -= seg; rem -= seg
            if rem > 0:
                seg = min(rem, y_h)
                if seg > 0:
                    self.create_rectangle(3, yb-seg, w-3, yb, fill=YELLOW, outline="")
                    yb -= seg; rem -= seg
            if rem > 0:
                self.create_rectangle(3, yb-rem, w-3, yb, fill=RED, outline="")

        pk_frac = max(0.0, (self._peak - DB_MIN) / (DB_MAX - DB_MIN))
        py = pt + bh - int(pk_frac * bh)
        pk_c = RED if self._peak > ZONE_RED else (YELLOW if self._peak > ZONE_YELLOW else GREEN)
        self.create_rectangle(3, py-2, w-3, py+2, fill=pk_c, outline="")

        for db_m in [0, -6, -12, -18, -30, -48, -60]:
            ym = pt + bh - int(max(0.0,(db_m-DB_MIN)/(DB_MAX-DB_MIN)) * bh)
            self.create_line(w-6, ym, w-2, ym, fill=MUTED)
        self.create_text(w//2, h-8, text=self.label, fill=MUTED, font=("Consolas", 8))


# ─────────────────────────────────────────────────────────────
# WYKRES HISTORII
# ─────────────────────────────────────────────────────────────

class HistoryChart(tk.Canvas):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG2, highlightthickness=0, height=100, **kw)
        self._hist = [DB_MIN] * HISTORY_LEN
        self.bind("<Configure>", lambda e: self._draw())

    def push(self, db):
        self._hist.append(db)
        if len(self._hist) > HISTORY_LEN:
            self._hist.pop(0)
        self._draw()

    def _draw(self):
        self.delete("all")
        w = self.winfo_width(); h = self.winfo_height()
        if w < 20 or h < 20:
            return
        pl=36; pr=6; pt=4; pb=16
        cw=w-pl-pr; ch=h-pt-pb
        self.create_rectangle(pl, pt, pl+cw, pt+ch, fill=BG3, outline=BORDER)
        for db_g in [0,-6,-12,-18,-30,-48,-60,-80]:
            frac = (db_g-DB_MIN)/(DB_MAX-DB_MIN)
            yg = pt+ch-int(frac*ch)
            self.create_line(pl, yg, pl+cw, yg, fill=BORDER, dash=(2,4))
            self.create_text(pl-3, yg, text=str(db_g), anchor=tk.E,
                             fill=MUTED, font=("Consolas", 7))
        n = len(self._hist)
        if n < 2:
            return
        step = cw / (HISTORY_LEN - 1)
        pts = []
        for i, db_v in enumerate(self._hist):
            frac = max(0.0, (db_v-DB_MIN)/(DB_MAX-DB_MIN))
            pts.extend([pl + i*step, pt+ch - frac*ch])
        self.create_line(*pts, fill=ACCENT, width=1.5, smooth=True)
        secs = int(HISTORY_LEN * BLOCK_SIZE / SAMPLE_RATE)
        self.create_text(pl+cw//2, h-4,
                         text=f"<-- ostatnie ~{secs} s",
                         fill=MUTED, font=("Consolas", 8))


# ─────────────────────────────────────────────────────────────
# PASEK POZIOMY
# ─────────────────────────────────────────────────────────────

class HBar(tk.Canvas):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG3, highlightthickness=0, height=22, **kw)
        self._db   = DB_MIN
        self._peak = DB_MIN
        self.bind("<Configure>", lambda e: self._draw())

    def set(self, db, peak):
        self._db = db; self._peak = peak; self._draw()

    def _draw(self):
        self.delete("all")
        w = self.winfo_width(); h = self.winfo_height()
        if w < 10: return
        rx = int(w*(ZONE_RED-DB_MIN)/(DB_MAX-DB_MIN))
        yx = int(w*(ZONE_YELLOW-DB_MIN)/(DB_MAX-DB_MIN))
        self.create_rectangle(0, 0, yx, h, fill="#14200e", outline="")
        self.create_rectangle(yx, 0, rx, h, fill="#1e1a08", outline="")
        self.create_rectangle(rx, 0, w, h, fill="#2a1010", outline="")
        frac = max(0.0, (self._db-DB_MIN)/(DB_MAX-DB_MIN))
        fx = int(frac*w)
        if fx > 0:
            gx = min(fx, yx)
            if gx > 0: self.create_rectangle(0, 2, gx, h-2, fill=GREEN, outline="")
            if fx > yx:
                yw2 = min(fx, rx)-yx
                if yw2 > 0: self.create_rectangle(yx, 2, yx+yw2, h-2, fill=YELLOW, outline="")
            if fx > rx: self.create_rectangle(rx, 2, fx, h-2, fill=RED, outline="")
        pkx = int(max(0.0,(self._peak-DB_MIN)/(DB_MAX-DB_MIN))*w)
        pkc = RED if self._peak>ZONE_RED else (YELLOW if self._peak>ZONE_YELLOW else GREEN)
        self.create_rectangle(pkx-2, 1, pkx+2, h-1, fill=pkc, outline="")


# ─────────────────────────────────────────────────────────────
# PANEL KLASYFIKACJI
# ─────────────────────────────────────────────────────────────

class PanelKlasyfikacji(tk.Frame):
    """
    Panel ktory wyswietla:
    - Aktualnie wykryta klase dzwieku (duzy napis)
    - Pewnosc w procentach
    - Paski dla wszystkich 6 klas
    - Historie ostatnich detekcji
    """
    def __init__(self, parent, klasy, **kw):
        super().__init__(parent, bg=BG2,
                         highlightbackground=BORDER, highlightthickness=1, **kw)
        self.klasy   = klasy
        self._historia = []  # ostatnie 8 detekcji

        # Naglowek
        tk.Label(self, text="KLASYFIKATOR DZWIEKOW",
                 font=("Consolas", 8), bg=BG2, fg=MUTED,
                 padx=8, pady=4).pack(anchor=tk.W)

        # Glowny wyswietlacz — wykryta klasa
        disp = tk.Frame(self, bg=BG2, padx=12, pady=8)
        disp.pack(fill=tk.X)

        tk.Label(disp, text="WYKRYTO:", font=("Consolas", 8),
                 bg=BG2, fg=MUTED).pack(anchor=tk.W)

        self.lbl_klasa = tk.Label(disp, text="— czekam na dzwiek —",
                                   font=("Consolas", 18, "bold"),
                                   bg=BG2, fg=MUTED)
        self.lbl_klasa.pack(anchor=tk.W)

        # Pasek pewnosci
        pewnosc_row = tk.Frame(disp, bg=BG2)
        pewnosc_row.pack(fill=tk.X, pady=4)
        tk.Label(pewnosc_row, text="Pewnosc:",
                 font=("Consolas", 9), bg=BG2, fg=MUTED).pack(side=tk.LEFT)
        self.lbl_pewnosc = tk.Label(pewnosc_row, text="—",
                                     font=("Consolas", 9, "bold"),
                                     bg=BG2, fg=ACCENT)
        self.lbl_pewnosc.pack(side=tk.LEFT, padx=6)

        # Separator
        tk.Frame(self, bg=BORDER, height=1).pack(fill=tk.X, padx=8)

        # Paski dla wszystkich klas
        tk.Label(self, text="PRAWDOPODOBIENSTWO KLAS:",
                 font=("Consolas", 8), bg=BG2, fg=MUTED,
                 padx=8, pady=6).pack(anchor=tk.W)

        self.paski = {}   # nazwa_klasy -> (label_proc, canvas_pasek)

        bars_frame = tk.Frame(self, bg=BG2, padx=8, pady=0)
        bars_frame.pack(fill=tk.X)

        for klasa in self.klasy:
            nazwa = NAZWY_PL.get(klasa, klasa)
            kolor = KOLORY_KLAS.get(klasa, ACCENT)

            row = tk.Frame(bars_frame, bg=BG2)
            row.pack(fill=tk.X, pady=2)

            # Nazwa klasy
            tk.Label(row, text=nazwa, font=("Consolas", 9),
                     bg=BG2, fg=TEXT, width=28, anchor=tk.W).pack(side=tk.LEFT)

            # Procent
            lbl_proc = tk.Label(row, text="  0%",
                                  font=("Consolas", 9, "bold"),
                                  bg=BG2, fg=kolor, width=5)
            lbl_proc.pack(side=tk.LEFT)

            # Pasek
            canvas = tk.Canvas(row, bg=BG3, highlightthickness=0,
                               height=14, width=200)
            canvas.pack(side=tk.LEFT, fill=tk.X, expand=True)
            canvas.bind("<Configure>",
                        lambda e, c=canvas, k=klasa: self._rysuj_pasek(c, k))

            self.paski[klasa] = (lbl_proc, canvas, kolor)

        # Separator
        tk.Frame(self, bg=BORDER, height=1).pack(fill=tk.X, padx=8, pady=6)

        # Historia detekcji
        tk.Label(self, text="HISTORIA DETEKCJI:",
                 font=("Consolas", 8), bg=BG2, fg=MUTED,
                 padx=8, pady=4).pack(anchor=tk.W)

        self.historia_frame = tk.Frame(self, bg=BG2, padx=8, pady=4)
        self.historia_frame.pack(fill=tk.X)

        self.lbl_historia = []
        for _ in range(6):
            lbl = tk.Label(self.historia_frame, text="",
                           font=("Consolas", 9), bg=BG2, fg=MUTED,
                           anchor=tk.W)
            lbl.pack(fill=tk.X)
            self.lbl_historia.append(lbl)

        # Aktualne wartosci (potrzebne do rysowania paskow)
        self._wartosci = {k: 0.0 for k in self.klasy}

    def aktualizuj(self, wynik):
        """
        Aktualizuje panel na podstawie wyniku klasyfikacji.
        wynik = {"klasa": ..., "pewnosc": ..., "wszystkie": {...}, "cisza": bool}
        """
        if wynik.get("cisza"):
            self.lbl_klasa.config(text="— cisza —", fg=MUTED)
            self.lbl_pewnosc.config(text="—")
            for klasa in self.klasy:
                self._wartosci[klasa] = 0.0
                self._aktualizuj_pasek(klasa, 0.0)
            return

        klasa   = wynik["klasa"]
        pewnosc = wynik["pewnosc"]
        wszystkie = wynik.get("wszystkie", {})

        # Duzy napis z nazwa klasy
        nazwa  = NAZWY_PL.get(klasa, klasa)
        kolor  = KOLORY_KLAS.get(klasa, ACCENT)
        self.lbl_klasa.config(text=nazwa, fg=kolor)

        # Pewnosc jako procent
        pct = int(pewnosc * 100)
        if pct >= 70:
            kolor_pct = GREEN
        elif pct >= 45:
            kolor_pct = YELLOW
        else:
            kolor_pct = RED
        self.lbl_pewnosc.config(text=f"{pct}%", fg=kolor_pct)

        # Aktualizuj paski dla wszystkich klas
        for k, v in wszystkie.items():
            self._wartosci[k] = v
            self._aktualizuj_pasek(k, v)

        # Dodaj do historii (tylko jesli pewnosc > 40%)
        if pewnosc >= 0.40:
            ts    = time.strftime("%H:%M:%S")
            wpis  = f"{ts}  {nazwa:<26}  {pct}%"
            self._historia.insert(0, (wpis, kolor))
            if len(self._historia) > 6:
                self._historia.pop()
            # Odswiez etykiety historii
            for i, lbl in enumerate(self.lbl_historia):
                if i < len(self._historia):
                    tekst, kol = self._historia[i]
                    lbl.config(text=tekst, fg=kol)
                else:
                    lbl.config(text="")

    def _aktualizuj_pasek(self, klasa, wartosc):
        if klasa not in self.paski:
            return
        lbl_proc, canvas, kolor = self.paski[klasa]
        pct = int(wartosc * 100)
        lbl_proc.config(text=f"{pct:3d}%")
        self._rysuj_pasek(canvas, klasa)

    def _rysuj_pasek(self, canvas, klasa):
        canvas.delete("all")
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w < 4 or h < 4:
            return
        canvas.create_rectangle(0, 0, w, h, fill=BG3, outline="")
        wartosc = self._wartosci.get(klasa, 0.0)
        fw = int(wartosc * w)
        if fw > 0:
            kolor = KOLORY_KLAS.get(klasa, ACCENT)
            canvas.create_rectangle(0, 2, fw, h-2, fill=kolor, outline="")


# ─────────────────────────────────────────────────────────────
# GLOWNA APLIKACJA
# ─────────────────────────────────────────────────────────────

class Aplikacja:
    def __init__(self, root):
        self.root        = root
        self.gui_q       = queue.Queue(maxsize=30)
        self.wynik_q     = queue.Queue(maxsize=10)

        # Laduj model
        self.model_pakiet  = zaladuj_model(MODEL_SCIEZKA)
        self.klasyfikator  = None
        self.audio_watek   = None
        self._uruchomiony  = False

        # Stan wyswietlacza
        self._db      = DB_MIN
        self._peak    = DB_MIN
        self._ch_dbs  = []
        self._min_db  = None
        self._max_db  = None
        self._probki  = 0

        self._buduj_ui()

        if self.model_pakiet:
            klasy = self.model_pakiet["klasy"]
            self.klasyfikator = KlasyfikatorWatek(self.model_pakiet, self.wynik_q)
        else:
            klasy = list(NAZWY_PL.keys())
            self._ustaw_status(
                f"UWAGA: Nie znaleziono modelu ({MODEL_SCIEZKA}). "
                "Wloz model_gotowy.pkl do tego samego folderu co skrypt.", YELLOW
            )

        # Przebuduj panel klasyfikacji z faktycznymi klasami
        self.panel_klas = PanelKlasyfikacji(self.prawa_kolumna, klasy)
        self.panel_klas.pack(fill=tk.BOTH, expand=True, pady=6)

        self._poll()

    # ── Budowanie UI ──────────────────────────────────────────

    def _buduj_ui(self):
        r = self.root
        r.title("Miernik Decybeli + Klasyfikator — Droniada 2026")
        r.configure(bg=BG)
        r.geometry("1100x720")
        r.minsize(900, 600)

        # Pasek gorny
        top = tk.Frame(r, bg="#0c1f33", padx=14, pady=8)
        top.pack(fill=tk.X)
        tk.Label(top, text="MIERNIK DECYBELI + KLASYFIKATOR",
                 font=("Consolas", 13, "bold"), bg="#0c1f33", fg=ACCENT).pack(side=tk.LEFT)
        tk.Label(top, text="  Droniada 2026",
                 font=("Consolas", 9), bg="#0c1f33", fg=MUTED).pack(side=tk.LEFT)
        self._dot     = tk.Label(top, text="●", font=("Consolas", 11),
                                  bg="#0c1f33", fg=MUTED)
        self._dot.pack(side=tk.RIGHT, padx=4)
        self._dot_lbl = tk.Label(top, text="Zatrzymany",
                                  font=("Consolas", 9), bg="#0c1f33", fg=MUTED)
        self._dot_lbl.pack(side=tk.RIGHT)

        # Konfiguracja
        cfg = tk.Frame(r, bg=BG2, padx=10, pady=8,
                       highlightbackground=BORDER, highlightthickness=1)
        cfg.pack(fill=tk.X, padx=6, pady=5)

        tk.Label(cfg, text="Urzadzenie:", font=("Consolas", 9),
                 bg=BG2, fg=MUTED).grid(row=0, column=0, sticky=tk.W, padx=0)

        self._devs    = self._pobierz_urzadzenia()
        dev_names     = [d[1] for d in self._devs]
        self._dev_var = tk.StringVar()
        self._dev_cb  = ttk.Combobox(cfg, textvariable=self._dev_var,
                                      values=dev_names or ["Brak urzadzen"],
                                      state="readonly", width=56, font=("Consolas", 9))
        self._dev_cb.grid(row=0, column=1, sticky=tk.W)
        if dev_names:
            domyslne = 0
            for k, (_, d) in enumerate(self._devs):
                if "mic" in d.lower() or "mikr" in d.lower():
                    domyslne = k; break
            self._dev_cb.current(domyslne)

        tk.Label(cfg, text="  Kanaly:", font=("Consolas", 9),
                 bg=BG2, fg=MUTED).grid(row=0, column=2, padx=14)
        self._ch_var = tk.IntVar(value=1)
        tk.Radiobutton(cfg, text="1 mono", variable=self._ch_var, value=1,
                       font=("Consolas", 9), bg=BG2, fg=TEXT,
                       selectcolor=BG3, activebackground=BG2
                       ).grid(row=0, column=3, padx=2)
        tk.Radiobutton(cfg, text="2 stereo", variable=self._ch_var, value=2,
                       font=("Consolas", 9), bg=BG2, fg=TEXT,
                       selectcolor=BG3, activebackground=BG2
                       ).grid(row=0, column=4, padx=2)

        # Model status
        model_info = f"Model: OK ({os.path.basename(MODEL_SCIEZKA)})" \
                     if self.model_pakiet else "Model: BRAK (tylko dB)"
        model_kolor = GREEN if self.model_pakiet else RED
        tk.Label(cfg, text=model_info, font=("Consolas", 9),
                 bg=BG2, fg=model_kolor).grid(row=0, column=5, padx=20)

        # Przyciski
        btn = tk.Frame(cfg, bg=BG2)
        btn.grid(row=1, column=0, columnspan=6, sticky=tk.W, pady=8)
        self._btn_start = tk.Button(btn, text="▶  Start",
                                     font=("Consolas", 10, "bold"),
                                     bg="#193d2c", fg=GREEN,
                                     activebackground="#2a5c40",
                                     relief=tk.FLAT, padx=14, pady=5,
                                     cursor="hand2", command=self._start)
        self._btn_start.pack(side=tk.LEFT, padx=0)
        self._btn_stop = tk.Button(btn, text="■  Stop",
                                    font=("Consolas", 10, "bold"),
                                    bg="#3a1616", fg=RED,
                                    activebackground="#5c2020",
                                    relief=tk.FLAT, padx=14, pady=5,
                                    state=tk.DISABLED, cursor="hand2",
                                    command=self._stop)
        self._btn_stop.pack(side=tk.LEFT, padx=0)
        tk.Button(btn, text="↺  Reset", font=("Consolas", 9),
                  bg=BG3, fg=MUTED, relief=tk.FLAT, padx=10, pady=5,
                  cursor="hand2", command=self._reset).pack(side=tk.LEFT)

        # Glowny obszar: lewo (VU + dB + wykres) | prawo (klasyfikator)
        main = tk.Frame(r, bg=BG)
        main.pack(fill=tk.BOTH, expand=True, padx=6, pady=5)

        # ── LEWA KOLUMNA ──────────────────────────────────────
        lewa = tk.Frame(main, bg=BG)
        lewa.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # VU + wyswietlacz
        top_row = tk.Frame(lewa, bg=BG)
        top_row.pack(fill=tk.X)

        # VU-metry
        self._vu_frame = tk.Frame(top_row, bg=BG2,
                                   highlightbackground=BORDER, highlightthickness=1,
                                   padx=8, pady=6)
        self._vu_frame.pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(self._vu_frame, text="VU", font=("Consolas", 8),
                 bg=BG2, fg=MUTED).pack()
        self._vu_inner = tk.Frame(self._vu_frame, bg=BG2)
        self._vu_inner.pack(fill=tk.BOTH, expand=True)
        self._vus = []
        self._zbuduj_vu(1)

        # Wyswietlacz liczbowy
        disp = tk.Frame(top_row, bg=BG2,
                        highlightbackground=BORDER, highlightthickness=1,
                        padx=14, pady=10)
        disp.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)

        row1 = tk.Frame(disp, bg=BG2)
        row1.pack(fill=tk.X)

        c1 = tk.Frame(row1, bg=BG2); c1.pack(side=tk.LEFT, padx=0)
        tk.Label(c1, text="POZIOM", font=("Consolas", 8),
                 bg=BG2, fg=MUTED).pack(anchor=tk.W)
        self._db_lbl = tk.Label(c1, text="-80.0 dB",
                                 font=("Consolas", 34, "bold"), bg=BG2, fg=GREEN)
        self._db_lbl.pack(anchor=tk.W)

        c2 = tk.Frame(row1, bg=BG2); c2.pack(side=tk.LEFT, padx=0)
        tk.Label(c2, text="PEAK", font=("Consolas", 8),
                 bg=BG2, fg=MUTED).pack(anchor=tk.W)
        self._peak_lbl = tk.Label(c2, text="-80.0 dB",
                                   font=("Consolas", 18, "bold"), bg=BG2, fg=YELLOW)
        self._peak_lbl.pack(anchor=tk.W)

        c3 = tk.Frame(row1, bg=BG2); c3.pack(side=tk.LEFT)
        tk.Label(c3, text="STATYSTYKI", font=("Consolas", 8),
                 bg=BG2, fg=MUTED).pack(anchor=tk.W)
        self._min_lbl = tk.Label(c3, text="Min:  —",
                                  font=("Consolas", 10), bg=BG2, fg=TEXT)
        self._min_lbl.pack(anchor=tk.W)
        self._max_lbl = tk.Label(c3, text="Max:  —",
                                  font=("Consolas", 10), bg=BG2, fg=TEXT)
        self._max_lbl.pack(anchor=tk.W)
        self._smp_lbl = tk.Label(c3, text="Probki: 0",
                                  font=("Consolas", 10), bg=BG2, fg=MUTED)
        self._smp_lbl.pack(anchor=tk.W)

        # Pasek poziomy
        self._hbar = HBar(disp)
        self._hbar.pack(fill=tk.X, pady=8)

        # Kanaly
        self._ch_frame = tk.Frame(disp, bg=BG2)
        self._ch_frame.pack(fill=tk.X, pady=6)
        self._ch_lbls = []
        self._zbuduj_ch_etykiety(1)

        # Wykres historii
        hist_frame = tk.Frame(lewa, bg=BG2,
                              highlightbackground=BORDER, highlightthickness=1)
        hist_frame.pack(fill=tk.BOTH, expand=True, pady=6)
        tk.Label(hist_frame, text="HISTORIA POZIOMU",
                 font=("Consolas", 8), bg=BG2, fg=MUTED,
                 padx=8, pady=3).pack(anchor=tk.W)
        self._chart = HistoryChart(hist_frame)
        self._chart.pack(fill=tk.BOTH, expand=True, padx=4, pady=0)

        # ── PRAWA KOLUMNA — klasyfikator (dodawana pozniej) ───
        self.prawa_kolumna = tk.Frame(main, bg=BG, width=380)
        self.prawa_kolumna.pack(side=tk.RIGHT, fill=tk.BOTH,
                                 padx=6)
        self.prawa_kolumna.pack_propagate(False)

        # Pasek statusu
        sb = tk.Frame(r, bg=BG3, padx=8, pady=3)
        sb.pack(fill=tk.X, side=tk.BOTTOM)
        self._status_lbl = tk.Label(sb, text="Gotowy. Wybierz urzadzenie i kliknij Start.",
                                     font=("Consolas", 8), bg=BG3, fg=MUTED, anchor=tk.W)
        self._status_lbl.pack(side=tk.LEFT, fill=tk.X)

        # ttk styl
        sty = ttk.Style()
        try: sty.theme_use("clam")
        except Exception: pass
        sty.configure("TCombobox", fieldbackground=BG3, background=BG3,
                       foreground=TEXT, selectbackground=BG2, arrowcolor=MUTED)

    def _pobierz_urzadzenia(self):
        if not SD_OK:
            return []
        try:
            devs = sd.query_devices()
            return [(i, f"[{i}] {d['name'][:50]}  ({d['max_input_channels']}ch)")
                    for i, d in enumerate(devs) if d["max_input_channels"] > 0]
        except Exception:
            return []

    def _zbuduj_vu(self, n):
        for w in self._vu_inner.winfo_children():
            w.destroy()
        self._vus = []
        for i in range(n):
            lbl = ["L","R","3","4"][i] if i < 4 else str(i+1)
            vu  = VUMeter(self._vu_inner, label=lbl)
            vu.pack(side=tk.LEFT, padx=2, fill=tk.Y, expand=True)
            self._vus.append(vu)

    def _zbuduj_ch_etykiety(self, n):
        for w in self._ch_frame.winfo_children():
            w.destroy()
        self._ch_lbls = []
        for i in range(n):
            lbl_n = ["L","R","3","4"][i] if i < 4 else str(i+1)
            col = tk.Frame(self._ch_frame, bg=BG2)
            col.pack(side=tk.LEFT, padx=0)
            tk.Label(col, text=f"Kanal {lbl_n}",
                     font=("Consolas", 8), bg=BG2, fg=MUTED).pack(anchor=tk.W)
            lbl = tk.Label(col, text="-80.0 dB",
                           font=("Consolas", 12, "bold"), bg=BG2, fg=GREEN)
            lbl.pack(anchor=tk.W)
            self._ch_lbls.append(lbl)

    # ── Sterowanie ────────────────────────────────────────────

    def _start(self):
        if not SD_OK:
            self._ustaw_status("BLAD: pip install sounddevice", RED); return

        sel = self._dev_cb.current()
        if sel < 0 or sel >= len(self._devs):
            self._ustaw_status("Wybierz urzadzenie!", YELLOW); return

        dev_idx = self._devs[sel][0]
        n_ch    = self._ch_var.get()

        self._reset()
        self._zbuduj_vu(n_ch)
        self._zbuduj_ch_etykiety(n_ch)

        # Uruchom klasyfikator jesli model zaladowany
        if self.klasyfikator:
            self.klasyfikator.start()

        self.audio_watek = AudioWatek(self.gui_q, self.klasyfikator)
        self.audio_watek.start(dev_idx, n_ch)

        self._uruchomiony = True
        self._btn_start.config(state=tk.DISABLED)
        self._btn_stop.config(state=tk.NORMAL)
        self._dot.config(fg=GREEN)
        self._dot_lbl.config(text="Nasluchuje...", fg=GREEN)
        self._ustaw_status(
            f"Aktywny: [{dev_idx}]  {n_ch}ch  |  "
            f"Klasyfikacja: {'TAK (co ' + str(KLASYFIKUJ_CO_SEK) + 's)' if self.klasyfikator else 'BRAK MODELU'}",
            ACCENT
        )

    def _stop(self):
        self._uruchomiony = False
        if self.audio_watek:
            self.audio_watek.stop()
        if self.klasyfikator:
            self.klasyfikator.stop()
        self._btn_start.config(state=tk.NORMAL)
        self._btn_stop.config(state=tk.DISABLED)
        self._dot.config(fg=MUTED)
        self._dot_lbl.config(text="Zatrzymany", fg=MUTED)
        self._ustaw_status("Zatrzymano.", MUTED)

    def _reset(self):
        self._min_db = None; self._max_db = None; self._probki = 0
        self._min_lbl.config(text="Min:  —")
        self._max_lbl.config(text="Max:  —")
        self._smp_lbl.config(text="Probki: 0")

    # ── Polling — serce aplikacji ─────────────────────────────

    def _poll(self):
        # Odbierz dane audio z kolejki
        zaktualizowano = False
        try:
            while True:
                msg = self.gui_q.get_nowait()
                if "error" in msg:
                    self._stop()
                    self._ustaw_status(f"Blad audio: {msg['error']}", RED)
                    break
                self._db     = msg["db"]
                self._peak   = msg["peak"]
                self._ch_dbs = msg["channels"]
                self._probki += 1
                if self._min_db is None or self._db < self._min_db:
                    self._min_db = self._db
                if self._max_db is None or self._db > self._max_db:
                    self._max_db = self._db
                zaktualizowano = True
        except queue.Empty:
            pass

        # Odbierz wynik klasyfikacji z kolejki
        try:
            while True:
                wynik = self.wynik_q.get_nowait()
                if hasattr(self, 'panel_klas'):
                    self.panel_klas.aktualizuj(wynik)
        except queue.Empty:
            pass

        # Odswiez wyswietlacz
        if zaktualizowano:
            self._odswiez()

        self.root.after(REFRESH_MS, self._poll)

    def _odswiez(self):
        db   = self._db
        peak = self._peak

        col = RED if db > ZONE_RED else (YELLOW if db > ZONE_YELLOW else GREEN)
        self._db_lbl.config(text=f"{db:+.1f} dB", fg=col)
        pk_c = RED if peak > ZONE_RED else YELLOW
        self._peak_lbl.config(text=f"{peak:+.1f} dB", fg=pk_c)

        mn = f"{self._min_db:+.1f} dB" if self._min_db is not None else "—"
        mx = f"{self._max_db:+.1f} dB" if self._max_db is not None else "—"
        self._min_lbl.config(text=f"Min:  {mn}", fg=ACCENT)
        self._max_lbl.config(text=f"Max:  {mx}",
                              fg=RED if self._max_db and self._max_db > ZONE_RED else YELLOW)
        self._smp_lbl.config(text=f"Probki: {self._probki}")

        for i, vu in enumerate(self._vus):
            vu.set(self._ch_dbs[i] if i < len(self._ch_dbs) else DB_MIN, peak)

        for i, lbl in enumerate(self._ch_lbls):
            v = self._ch_dbs[i] if i < len(self._ch_dbs) else DB_MIN
            c = RED if v > ZONE_RED else (YELLOW if v > ZONE_YELLOW else GREEN)
            lbl.config(text=f"{v:+.1f} dB", fg=c)

        self._hbar.set(db, peak)
        self._chart.push(db)

    def _ustaw_status(self, txt, kol=MUTED):
        self._status_lbl.config(text=txt, fg=kol)

    def on_close(self):
        self._stop()
        self.root.destroy()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    sty  = ttk.Style(root)
    try:
        sty.theme_use("clam")
    except Exception:
        pass
    sty.configure("TCombobox", fieldbackground=BG3, background=BG3,
                   foreground=TEXT, selectbackground=BG2, arrowcolor=MUTED)

    app = Aplikacja(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
