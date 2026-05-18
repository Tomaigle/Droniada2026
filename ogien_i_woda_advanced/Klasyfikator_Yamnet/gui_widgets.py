import tkinter as tk
import time
from config import *

class VUMeter(tk.Canvas):
    def __init__(self, parent, label="L", **kw):
        super().__init__(parent, bg=BG2, highlightthickness=0, width=36, height=260, **kw)
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
        if w < 4 or h < 4: return
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
        if w < 20 or h < 20: return
        pl=36; pr=6; pt=4; pb=16
        cw=w-pl-pr; ch=h-pt-pb
        self.create_rectangle(pl, pt, pl+cw, pt+ch, fill=BG3, outline=BORDER)
        for db_g in [0,-6,-12,-18,-30,-48,-60,-80]:
            frac = (db_g-DB_MIN)/(DB_MAX-DB_MIN)
            yg = pt+ch-int(frac*ch)
            self.create_line(pl, yg, pl+cw, yg, fill=BORDER, dash=(2,4))
            self.create_text(pl-3, yg, text=str(db_g), anchor=tk.E, fill=MUTED, font=("Consolas", 7))
        n = len(self._hist)
        if n < 2: return
        step = cw / (HISTORY_LEN - 1)
        pts = []
        for i, db_v in enumerate(self._hist):
            frac = max(0.0, (db_v-DB_MIN)/(DB_MAX-DB_MIN))
            pts.extend([pl + i*step, pt+ch - frac*ch])
        self.create_line(*pts, fill=ACCENT, width=1.5, smooth=True)
        secs = int(HISTORY_LEN * BLOCK_SIZE / SAMPLE_RATE)
        self.create_text(pl+cw//2, h-4, text=f"<-- ostatnie ~{secs} s", fill=MUTED, font=("Consolas", 8))

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

class PanelKlasyfikacji(tk.Frame):
    def __init__(self, parent, klasy, **kw):
        super().__init__(parent, bg=BG2, highlightbackground=BORDER, highlightthickness=1, **kw)
        self.klasy   = klasy
        self._historia = []

        tk.Label(self, text="KLASYFIKATOR DZWIEKOW", font=("Consolas", 8), bg=BG2, fg=MUTED, padx=8, pady=4).pack(anchor=tk.W)

        disp = tk.Frame(self, bg=BG2, padx=12, pady=8)
        disp.pack(fill=tk.X)
        tk.Label(disp, text="WYKRYTO:", font=("Consolas", 8), bg=BG2, fg=MUTED).pack(anchor=tk.W)
        self.lbl_klasa = tk.Label(disp, text="— czekam na dzwiek —", font=("Consolas", 18, "bold"), bg=BG2, fg=MUTED)
        self.lbl_klasa.pack(anchor=tk.W)

        pewnosc_row = tk.Frame(disp, bg=BG2)
        pewnosc_row.pack(fill=tk.X, pady=4)
        tk.Label(pewnosc_row, text="Pewnosc:", font=("Consolas", 9), bg=BG2, fg=MUTED).pack(side=tk.LEFT)
        self.lbl_pewnosc = tk.Label(pewnosc_row, text="—", font=("Consolas", 9, "bold"), bg=BG2, fg=ACCENT)
        self.lbl_pewnosc.pack(side=tk.LEFT, padx=6)

        tk.Frame(self, bg=BORDER, height=1).pack(fill=tk.X, padx=8)
        tk.Label(self, text="PRAWDOPODOBIENSTWO KLAS:", font=("Consolas", 8), bg=BG2, fg=MUTED, padx=8, pady=6).pack(anchor=tk.W)

        self.paski = {}
        bars_frame = tk.Frame(self, bg=BG2, padx=8, pady=0)
        bars_frame.pack(fill=tk.X)

        for klasa in self.klasy:
            nazwa = NAZWY_PL.get(klasa, klasa)
            kolor = KOLORY_KLAS.get(klasa, ACCENT)

            row = tk.Frame(bars_frame, bg=BG2)
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=nazwa, font=("Consolas", 9), bg=BG2, fg=TEXT, width=28, anchor=tk.W).pack(side=tk.LEFT)

            lbl_proc = tk.Label(row, text="  0%", font=("Consolas", 9, "bold"), bg=BG2, fg=kolor, width=5)
            lbl_proc.pack(side=tk.LEFT)

            canvas = tk.Canvas(row, bg=BG3, highlightthickness=0, height=14, width=200)
            canvas.pack(side=tk.LEFT, fill=tk.X, expand=True)
            canvas.bind("<Configure>", lambda e, c=canvas, k=klasa: self._rysuj_pasek(c, k))

            self.paski[klasa] = (lbl_proc, canvas, kolor)

        tk.Frame(self, bg=BORDER, height=1).pack(fill=tk.X, padx=8, pady=6)
        tk.Label(self, text="HISTORIA DETEKCJI:", font=("Consolas", 8), bg=BG2, fg=MUTED, padx=8, pady=4).pack(anchor=tk.W)

        self.historia_frame = tk.Frame(self, bg=BG2, padx=8, pady=4)
        self.historia_frame.pack(fill=tk.X)

        self.lbl_historia = []
        for _ in range(6):
            lbl = tk.Label(self.historia_frame, text="", font=("Consolas", 9), bg=BG2, fg=MUTED, anchor=tk.W)
            lbl.pack(fill=tk.X)
            self.lbl_historia.append(lbl)

        self._wartosci = {k: 0.0 for k in self.klasy}

    def aktualizuj(self, wynik):
        if wynik.get("cisza"):
            self.lbl_klasa.config(text="— czekam na dzwiek —", fg=MUTED)
            self.lbl_pewnosc.config(text="—")
            for klasa in self.klasy:
                self._wartosci[klasa] = 0.0
                self._aktualizuj_pasek(klasa, 0.0)
            return

        klasa   = wynik["klasa"]
        pewnosc = wynik["pewnosc"]
        wszystkie = wynik.get("wszystkie", {})

        nazwa  = NAZWY_PL.get(klasa, klasa)
        kolor  = KOLORY_KLAS.get(klasa, ACCENT)
        self.lbl_klasa.config(text=nazwa, fg=kolor)

        pct = int(pewnosc * 100)
        kolor_pct = GREEN if pct >= 70 else (YELLOW if pct >= 45 else RED)
        self.lbl_pewnosc.config(text=f"{pct}%", fg=kolor_pct)

        for k, v in wszystkie.items():
            self._wartosci[k] = v
            self._aktualizuj_pasek(k, v)

        if pewnosc >= 0.40:
            ts    = time.strftime("%H:%M:%S")
            wpis  = f"{ts}  {nazwa:<26}  {pct}%"
            self._historia.insert(0, (wpis, kolor))
            if len(self._historia) > 6:
                self._historia.pop()
            for i, lbl in enumerate(self.lbl_historia):
                if i < len(self._historia):
                    tekst, kol = self._historia[i]
                    lbl.config(text=tekst, fg=kol)
                else:
                    lbl.config(text="")

    def _aktualizuj_pasek(self, klasa, wartosc):
        if klasa not in self.paski: return
        lbl_proc, canvas, kolor = self.paski[klasa]
        pct = int(wartosc * 100)
        lbl_proc.config(text=f"{pct:3d}%")
        self._rysuj_pasek(canvas, klasa)

    def _rysuj_pasek(self, canvas, klasa):
        canvas.delete("all")
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w < 4 or h < 4: return
        canvas.create_rectangle(0, 0, w, h, fill=BG3, outline="")
        wartosc = self._wartosci.get(klasa, 0.0)
        fw = int(wartosc * w)
        if fw > 0:
            kolor = KOLORY_KLAS.get(klasa, ACCENT)
            canvas.create_rectangle(0, 2, fw, h-2, fill=kolor, outline="")
