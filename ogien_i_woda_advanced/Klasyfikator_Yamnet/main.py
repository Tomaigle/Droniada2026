import os
import tkinter as tk
from tkinter import ttk
import queue

from config import *
from audio_engine import AudioWatek
from ai_engine import zaladuj_model, KlasyfikatorWatek
from gui_widgets import VUMeter, HistoryChart, HBar, PanelKlasyfikacji

if SD_OK:
    import sounddevice as sd

class Aplikacja:
    def __init__(self, root):
        self.root        = root
        self.gui_q       = queue.Queue(maxsize=30)
        self.wynik_q     = queue.Queue(maxsize=10)

        # Laduj model z modulu AI
        self.model_pakiet = zaladuj_model()
        self.klasyfikator  = None
        self.audio_watek   = None
        self._uruchomiony  = False

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
                f"UWAGA: Nie znaleziono modelu ({os.path.basename(MODEL_KERAS_SCIEZKA)}).", YELLOW
            )

        self.panel_klas = PanelKlasyfikacji(self.prawa_kolumna, klasy)
        self.panel_klas.pack(fill=tk.BOTH, expand=True, pady=6)

        self._poll()

    def _buduj_ui(self):
        r = self.root
        r.title("Miernik Decybeli + Klasyfikator — OOP Refactored")
        r.configure(bg=BG)
        r.geometry("1100x720")
        r.minsize(900, 600)

        top = tk.Frame(r, bg="#0c1f33", padx=14, pady=8)
        top.pack(fill=tk.X)
        tk.Label(top, text="MIERNIK DECYBELI + KLASYFIKATOR", font=("Consolas", 13, "bold"), bg="#0c1f33", fg=ACCENT).pack(side=tk.LEFT)
        tk.Label(top, text="  Droniada 2026", font=("Consolas", 9), bg="#0c1f33", fg=MUTED).pack(side=tk.LEFT)
        self._dot     = tk.Label(top, text="●", font=("Consolas", 11), bg="#0c1f33", fg=MUTED)
        self._dot.pack(side=tk.RIGHT, padx=4)
        self._dot_lbl = tk.Label(top, text="Zatrzymany", font=("Consolas", 9), bg="#0c1f33", fg=MUTED)
        self._dot_lbl.pack(side=tk.RIGHT)

        cfg = tk.Frame(r, bg=BG2, padx=10, pady=8, highlightbackground=BORDER, highlightthickness=1)
        cfg.pack(fill=tk.X, padx=6, pady=5)

        tk.Label(cfg, text="Urzadzenie:", font=("Consolas", 9), bg=BG2, fg=MUTED).grid(row=0, column=0, sticky=tk.W, padx=0)

        self._devs    = self._pobierz_urzadzenia()
        dev_names     = [d[1] for d in self._devs]
        self._dev_var = tk.StringVar()
        self._dev_cb  = ttk.Combobox(cfg, textvariable=self._dev_var, values=dev_names or ["Brak urzadzen"], state="readonly", width=56, font=("Consolas", 9))
        self._dev_cb.grid(row=0, column=1, sticky=tk.W)
        if dev_names:
            domyslne = 0
            for k, (_, d) in enumerate(self._devs):
                if "mic" in d.lower() or "mikr" in d.lower():
                    domyslne = k; break
            self._dev_cb.current(domyslne)

        tk.Label(cfg, text="  Kanaly:", font=("Consolas", 9), bg=BG2, fg=MUTED).grid(row=0, column=2, padx=14)
        self._ch_var = tk.IntVar(value=1)
        tk.Radiobutton(cfg, text="1 mono", variable=self._ch_var, value=1, font=("Consolas", 9), bg=BG2, fg=TEXT, selectcolor=BG3, activebackground=BG2).grid(row=0, column=3, padx=2)
        tk.Radiobutton(cfg, text="2 stereo", variable=self._ch_var, value=2, font=("Consolas", 9), bg=BG2, fg=TEXT, selectcolor=BG3, activebackground=BG2).grid(row=0, column=4, padx=2)

        model_info = f"Model: OK ({os.path.basename(MODEL_KERAS_SCIEZKA)})" if self.model_pakiet else "Model: BRAK (tylko dB)"
        model_kolor = GREEN if self.model_pakiet else RED
        tk.Label(cfg, text=model_info, font=("Consolas", 9), bg=BG2, fg=model_kolor).grid(row=0, column=5, padx=20)

        btn = tk.Frame(cfg, bg=BG2)
        btn.grid(row=1, column=0, columnspan=6, sticky=tk.W, pady=8)
        self._btn_start = tk.Button(btn, text="▶  Start", font=("Consolas", 10, "bold"), bg="#193d2c", fg=GREEN, activebackground="#2a5c40", relief=tk.FLAT, padx=14, pady=5, cursor="hand2", command=self._start)
        self._btn_start.pack(side=tk.LEFT, padx=0)
        self._btn_stop = tk.Button(btn, text="■  Stop", font=("Consolas", 10, "bold"), bg="#3a1616", fg=RED, activebackground="#5c2020", relief=tk.FLAT, padx=14, pady=5, state=tk.DISABLED, cursor="hand2", command=self._stop)
        self._btn_stop.pack(side=tk.LEFT, padx=0)
        tk.Button(btn, text="↺  Reset", font=("Consolas", 9), bg=BG3, fg=MUTED, relief=tk.FLAT, padx=10, pady=5, cursor="hand2", command=self._reset).pack(side=tk.LEFT)

        main = tk.Frame(r, bg=BG)
        main.pack(fill=tk.BOTH, expand=True, padx=6, pady=5)

        lewa = tk.Frame(main, bg=BG)
        lewa.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        top_row = tk.Frame(lewa, bg=BG)
        top_row.pack(fill=tk.X)

        self._vu_frame = tk.Frame(top_row, bg=BG2, highlightbackground=BORDER, highlightthickness=1, padx=8, pady=6)
        self._vu_frame.pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(self._vu_frame, text="VU", font=("Consolas", 8), bg=BG2, fg=MUTED).pack()
        self._vu_inner = tk.Frame(self._vu_frame, bg=BG2)
        self._vu_inner.pack(fill=tk.BOTH, expand=True)
        self._vus = []
        self._zbuduj_vu(1)

        disp = tk.Frame(top_row, bg=BG2, highlightbackground=BORDER, highlightthickness=1, padx=14, pady=10)
        disp.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)

        row1 = tk.Frame(disp, bg=BG2)
        row1.pack(fill=tk.X)

        c1 = tk.Frame(row1, bg=BG2); c1.pack(side=tk.LEFT, padx=0)
        tk.Label(c1, text="POZIOM", font=("Consolas", 8), bg=BG2, fg=MUTED).pack(anchor=tk.W)
        self._db_lbl = tk.Label(c1, text="-80.0 dB", font=("Consolas", 34, "bold"), bg=BG2, fg=GREEN)
        self._db_lbl.pack(anchor=tk.W)

        c2 = tk.Frame(row1, bg=BG2); c2.pack(side=tk.LEFT, padx=0)
        tk.Label(c2, text="PEAK", font=("Consolas", 8), bg=BG2, fg=MUTED).pack(anchor=tk.W)
        self._peak_lbl = tk.Label(c2, text="-80.0 dB", font=("Consolas", 18, "bold"), bg=BG2, fg=YELLOW)
        self._peak_lbl.pack(anchor=tk.W)

        c3 = tk.Frame(row1, bg=BG2); c3.pack(side=tk.LEFT)
        tk.Label(c3, text="STATYSTYKI", font=("Consolas", 8), bg=BG2, fg=MUTED).pack(anchor=tk.W)
        self._min_lbl = tk.Label(c3, text="Min:  —", font=("Consolas", 10), bg=BG2, fg=TEXT)
        self._min_lbl.pack(anchor=tk.W)
        self._max_lbl = tk.Label(c3, text="Max:  —", font=("Consolas", 10), bg=BG2, fg=TEXT)
        self._max_lbl.pack(anchor=tk.W)
        self._smp_lbl = tk.Label(c3, text="Probki: 0", font=("Consolas", 10), bg=BG2, fg=MUTED)
        self._smp_lbl.pack(anchor=tk.W)

        self._hbar = HBar(disp)
        self._hbar.pack(fill=tk.X, pady=8)

        self._ch_frame = tk.Frame(disp, bg=BG2)
        self._ch_frame.pack(fill=tk.X, pady=6)
        self._ch_lbls = []
        self._zbuduj_ch_etykiety(1)

        hist_frame = tk.Frame(lewa, bg=BG2, highlightbackground=BORDER, highlightthickness=1)
        hist_frame.pack(fill=tk.BOTH, expand=True, pady=6)
        tk.Label(hist_frame, text="HISTORIA POZIOMU", font=("Consolas", 8), bg=BG2, fg=MUTED, padx=8, pady=3).pack(anchor=tk.W)
        self._chart = HistoryChart(hist_frame)
        self._chart.pack(fill=tk.BOTH, expand=True, padx=4, pady=0)

        self.prawa_kolumna = tk.Frame(main, bg=BG, width=380)
        self.prawa_kolumna.pack(side=tk.RIGHT, fill=tk.BOTH, padx=6)
        self.prawa_kolumna.pack_propagate(False)

        sb = tk.Frame(r, bg=BG3, padx=8, pady=3)
        sb.pack(fill=tk.X, side=tk.BOTTOM)
        self._status_lbl = tk.Label(sb, text="Gotowy. Wybierz urzadzenie i kliknij Start.", font=("Consolas", 8), bg=BG3, fg=MUTED, anchor=tk.W)
        self._status_lbl.pack(side=tk.LEFT, fill=tk.X)

    def _pobierz_urzadzenia(self):
        if not SD_OK: return []
        try:
            devs = sd.query_devices()
            return [(i, f"[{i}] {d['name'][:50]}  ({d['max_input_channels']}ch)") for i, d in enumerate(devs) if d["max_input_channels"] > 0]
        except Exception: return []

    def _zbuduj_vu(self, n):
        for w in self._vu_inner.winfo_children(): w.destroy()
        self._vus = []
        for i in range(n):
            lbl = ["L","R","3","4"][i] if i < 4 else str(i+1)
            vu  = VUMeter(self._vu_inner, label=lbl)
            vu.pack(side=tk.LEFT, padx=2, fill=tk.Y, expand=True)
            self._vus.append(vu)

    def _zbuduj_ch_etykiety(self, n):
        for w in self._ch_frame.winfo_children(): w.destroy()
        self._ch_lbls = []
        for i in range(n):
            lbl_n = ["L","R","3","4"][i] if i < 4 else str(i+1)
            col = tk.Frame(self._ch_frame, bg=BG2)
            col.pack(side=tk.LEFT, padx=0)
            tk.Label(col, text=f"Kanal {lbl_n}", font=("Consolas", 8), bg=BG2, fg=MUTED).pack(anchor=tk.W)
            lbl = tk.Label(col, text="-80.0 dB", font=("Consolas", 12, "bold"), bg=BG2, fg=GREEN)
            lbl.pack(anchor=tk.W)
            self._ch_lbls.append(lbl)

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

        if self.klasyfikator: self.klasyfikator.start()
        self.audio_watek = AudioWatek(self.gui_q, self.klasyfikator)
        self.audio_watek.start(dev_idx, n_ch)

        self._uruchomiony = True
        self._btn_start.config(state=tk.DISABLED)
        self._btn_stop.config(state=tk.NORMAL)
        self._dot.config(fg=GREEN)
        self._dot_lbl.config(text="Nasluchuje...", fg=GREEN)
        self._ustaw_status(f"Aktywny: [{dev_idx}]  {n_ch}ch  |  Klasyfikacja: {'TAK (co ' + str(KLASYFIKUJ_CO_SEK) + 's)' if self.klasyfikator else 'BRAK MODELU'}", ACCENT)

    def _stop(self):
        self._uruchomiony = False
        if self.audio_watek: self.audio_watek.stop()
        if self.klasyfikator: self.klasyfikator.stop()
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

    def _poll(self):
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
                if self._min_db is None or self._db < self._min_db: self._min_db = self._db
                if self._max_db is None or self._db > self._max_db: self._max_db = self._db
                zaktualizowano = True
        except queue.Empty: pass

        try:
            while True:
                wynik = self.wynik_q.get_nowait()
                if hasattr(self, 'panel_klas'): self.panel_klas.aktualizuj(wynik)
        except queue.Empty: pass

        if zaktualizowano: self._odswiez()
        self.root.after(REFRESH_MS, self._poll)

    def _odswiez(self):
        db, peak = self._db, self._peak
        col = RED if db > ZONE_RED else (YELLOW if db > ZONE_YELLOW else GREEN)
        self._db_lbl.config(text=f"{db:+.1f} dB", fg=col)
        pk_c = RED if peak > ZONE_RED else YELLOW
        self._peak_lbl.config(text=f"{peak:+.1f} dB", fg=pk_c)

        mn = f"{self._min_db:+.1f} dB" if self._min_db is not None else "—"
        mx = f"{self._max_db:+.1f} dB" if self._max_db is not None else "—"
        self._min_lbl.config(text=f"Min:  {mn}", fg=ACCENT)
        self._max_lbl.config(text=f"Max:  {mx}", fg=RED if self._max_db and self._max_db > ZONE_RED else YELLOW)
        self._smp_lbl.config(text=f"Probki: {self._probki}")

        for i, vu in enumerate(self._vus): vu.set(self._ch_dbs[i] if i < len(self._ch_dbs) else DB_MIN, peak)
        for i, lbl in enumerate(self._ch_lbls):
            v = self._ch_dbs[i] if i < len(self._ch_dbs) else DB_MIN
            lbl.config(text=f"{v:+.1f} dB", fg=RED if v > ZONE_RED else (YELLOW if v > ZONE_YELLOW else GREEN))

        self._hbar.set(db, peak)
        self._chart.push(db)

    def _ustaw_status(self, txt, kol=MUTED):
        self._status_lbl.config(text=txt, fg=kol)

    def on_close(self):
        self._stop()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    sty  = ttk.Style(root)
    try: sty.theme_use("clam")
    except Exception: pass
    sty.configure("TCombobox", fieldbackground=BG3, background=BG3, foreground=TEXT, selectbackground=BG2, arrowcolor=MUTED)

    app = Aplikacja(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
