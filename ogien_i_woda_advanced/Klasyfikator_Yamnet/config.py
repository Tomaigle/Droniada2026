import os

try:
    import sounddevice as sd
    SD_OK = True
except ImportError:
    SD_OK = False

# ─────────────────────────────────────────────────────────────
# STALE AUDIO
# ─────────────────────────────────────────────────────────────
SAMPLE_RATE   = 48000
BLOCK_SIZE    = 4096
HISTORY_LEN   = 200
DB_MIN        = -80.0
DB_MAX        = 0.0
ZONE_YELLOW   = -6.0
ZONE_RED      = -3.0

# ─────────────────────────────────────────────────────────────
# STALE AI
# ─────────────────────────────────────────────────────────────
KLASYFIKUJ_CO_SEK = 3.0
MODEL_KERAS_SCIEZKA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moj_model.keras")
KLASY_PKL_SCIEZKA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "klasy.pkl")

# ─────────────────────────────────────────────────────────────
# WYGLAD (GUI)
# ─────────────────────────────────────────────────────────────
REFRESH_MS    = 60

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

KOLORY_KLAS = {
    "bomba_lotnicza":         "#f85149",
    "kolumna_pancerna":       "#d29922",
    "pozar_trzask":           "#ff7b00",
    "serie_broni_maszynowej": "#a371f7",
    "woda_wyciek":            "#58a6ff",
    "wystrzal_krab":          "#f0883e",
    "szumy":                  "#8b949e", # Dodane szumy
}

NAZWY_PL = {
    "bomba_lotnicza":         "Bomba lotnicza",
    "kolumna_pancerna":       "Kolumna pancerna",
    "pozar_trzask":           "Pozar (trzask)",
    "serie_broni_maszynowej": "Serie z broni maszynowej",
    "woda_wyciek":            "Woda (wyciek)",
    "wystrzal_krab":          "Wystrzal Krab",
    "szumy":                  "Szum z laptopa", # Dodane szumy
}
