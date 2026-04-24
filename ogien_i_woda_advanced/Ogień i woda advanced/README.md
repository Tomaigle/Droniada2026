# 🎯 Sound Classifier GUI v1.1 - README

## 📌 Co To Jest?

Zaawansowany system wykrywania i klasyfikacji dźwięków bojowych w **realtimie**:
- 🎙️ Nagrywanie z mikrofonu na żywo
- 🤖 Sztuczna inteligencja (ONNX/YAMNet) + heurystyka
- 📊 GUI graficzny z WU metrem i wykresami
- 📁 Automatyczne raporty JSON z numeracją
- 🎯 6 kategorii dźwięków (bomby, serie broni, kolumny pancerne itd.)

---

## 🚀 Szybki Start (3 kroki)

### 1️⃣ Zainstaluj Zależności
```bash
pip install librosa sounddevice numpy scipy onnxruntime
```

### 2️⃣ Umieść Model AI
- Pobierz: `yamnet_classifier.onnx`
- Umieść w folderze obok skryptu

### 3️⃣ Uruchom
```bash
python Sound_classifier_gui2.py
```

---

## 📋 Nowoości w v1.1 ✨

Poprawiłem **3 główne rzeczy**:

### 1️⃣ Lepsza Detekcja Dźwięków ✓
```
Przed:  Nyły proste reguły (3 warunki)
Po:     5 zaawansowanych reguł + hybrid AI
        MIN_CONFIDENCE: 0.35 → 0.25 (bardziej czuły)
```

**Co się zmienia?**
- ✨ Więcej parametrów akustycznych (ZCR, Spectral Centroid)
- ✨ Dynamiczne współczynniki confidence
- ✨ Fallback+AI współpracują razem
- ✨ Niska czułość na szumy tła

### 2️⃣ Wyświetlanie w GUI ✓
```
Przed:  [conf=0.45]  kurt=2.3
Po:     [conf=45%] RMS=-15.2dB  Kurt=2.3  ZCR=0.045  SC=1200Hz
        ↑ Procenty  ↑ dB      ↑ Cztery dodatkowe parametry
```

**Nowe informacje:**
- 📊 Confidence w **procentach** (45% zamiast 0.45)
- 🔊 **RMS w decybelach** (łatwiej czytać)
- ⚡ **Kurt** (Kurtosis - "ostość" dźwięku, detektuje impulsy)
- 🌊 **ZCR** (Zero Crossing Rate - szybkość zmian)
- 🎼 **SC** (Spectral Centroid - gdzie w spektrum)

### 3️⃣ Raporty JSON z Datą i Numeracją ✓
```
Przed:  report_20260405_143215.json      ← Problem: duplikaty!
Po:     report_20260405_143215_001.json
        report_20260405_143215_002.json  ← Numeracja + Polski format
```

**Nowy format JSON:**
```json
{
  "timestamp": "2026-04-05T14:32:15.123456",    ← ISO
  "datetime_pl": "2026-04-05 14:32:15",         ← Polski
  "duration_sec": 360,                          ← Nowe
  "sample_rate": 16000,                         ← Nowe
  "all_detections": {...}                       ← Nowe (all, nie tylko top3)
}
```

---

## 📂 Struktura Plików

```
Programowanie/Python/
├── Sound_classifier_gui2.py          ← GŁÓWNY PROGRAM
├── yamnet_classifier.onnx            ← Model AI (pobierz online)
├── akustykaai.py                     ← Starszy skrypt (do usunięcia?)
├── test.py                           ← Testy (jeśli są)
├── ZMIANY_I_INSTRUKCJE.md            ← Dokumentacja zmian
├── PORADNIK_OPTYMALIZACJI.md         ← Poradnik dostrojenia
├── CHANGELOG.md                      ← Historia wersji
├── PRZYKLAD_RAPORT.json              ← Przykład raportu
└── TESTER_V1.1.py                    ← Weryfiej instalację
```

---

## 🎮 Interfejs GUI

### Góra: Ustawienia
- 🎤 **Wybór mikrofonu** + Test
- ⏱️ **Czas testu** (presets: 30s, 60s, 110s, 360s)
- 📊 **VU Meter** (pasek poziomu audio w dB)

### Środek: Pasek Postępu
- ⏱️ **Zegar** (00:00 / 06:00)
- 🟢 **Status** (STAND-BY / NASLUCH AKTYWNY)
- ▓░ **Pasek postępu**

### Przyciski
- ▶️ **START** - Rozpocznij misję
- ⏹️ **STOP** - Zatrzymaj (zapisze raport)

### Dół: Wyniki (side-by-side)
- **LOG** (po lewej) - Lista wszystkich detekcji
  ```
  [14:32:15] ~ Woda / wyciek        [conf=67%] RMS=−15.2dB ...
  [14:32:16] # Pozar / trzask       [conf=58%] RMS=−12.1dB ...
  ```

- **WYNIKI** (po prawej) - Paski progresji
  ```
  ~ Woda / wyciek      [████░░░░░░░] 5 detekcji | score: 12.3
  # Pozar / trzask     [███░░░░░░░░░] 8 detekcji | score: 18.7
  = Kolumna pancerna   [███░░░░░░░░░] 3 detekcji | score: 8.2
  ```

---

## 🎯 Kategorie Dźwięków

| Ikona | Kategoria | Typ | Opis |
|-------|-----------|-----|------|
| ~ | Woda/wyciek | Ciągły | Szmer wody, wyciek |
| # | Pożar/trzask | Ciągły | Trzask drewna, pożar |
| ! | Wystrzał KRAB | Przerywany | Pojedynczy strzał |
| \* | Bomba lotnicza | Przerywany | Eksplozja |
| = | Kolumna pancerna | Ciągły | Czołg, silnik |
| : | Serie broni | Przerywany | Seria strzałów |

---

## ⚙️ Dostrojenie Parametrów

### Jeśli System Nie Detektuje

**Opcja 1: Obniż próg pewności**
```python
# W GUi2.py, linia ~32:
MIN_CONFIDENCE = 0.20  # Zamiast 0.25
```

**Opcja 2: Zwiększ czułość na cichość**
```python
SILENCE_RMS = 0.005  # Zamiast 0.008
```

**Opcja 3: Zmniejsz okno analizy** (szybsza, mniej stabilna)
```python
WINDOW_SEC = 1.5    # Zamiast 2.0
```

### Jeśli Zbyt Wiele Fałszywych Alarmów

**Zwiększ próg pewności:**
```python
MIN_CONFIDENCE = 0.40  # Zamiast 0.25
```

---

## 🧪 Testowanie

### Test 1: Weryfikuj Instalację
```bash
python TESTER_V1.1.py
```
Powinien pokazać zielone ✓ dla każdego testu.

### Test 2: Test Mikrofonu
1. Uruchom program
2. Kliknij **"TEST MIKROFONU"**
3. Powiedz coś - VU meter powinien się ruszać
4. Kliknij ponownie aby zatrzymać

### Test 3: Rzeczywista Detekcja
1. Kliknij **START**
2. Wybierz czas (60 sekund)
3. Wydaj dźwięki (klap, knaję, szeptem)
4. Obserwuj LOG - powinny się pojawić detekcje
5. Kliknij **STOP** - powinna się wykonać analiza

---

## 📊 Czytanie Raportów JSON

```bash
# Wyświetl raport (jeśli masz cat/type)
cat report_20260405_143215_001.json

# Lub otwórz w VS Code
code report_20260405_143215_001.json
```

**Opis pól:**

```json
{
  "timestamp": "2026-04-05T14:32:15.123456",     // ISO format (międzynarodowy)
  "datetime_pl": "2026-04-05 14:32:15",          // Polski format (czytelny)
  "duration_sec": 360,                           // Jak długo nagrywaliśmy
  "sample_rate": 16000,                          // Próbkowanie (Hz)
  
  "top3": [                                      // TOP 3 wyniki
    {
      "rank": 1,
      "category": "bomba_lotnicza",              // ID kategorii
      "label": "Bomba lotnicza",                 // Polska nazwa
      "type": "przerywany",                     // CIĄGŁY/PRZERYWANY
      "score": 52.5,                             // Suma confidence (więcej = lepiej)
      "detections": 28                           // Ile razy wykrytą
    },
    // rank 2 i 3...
  ],
  
  "all_scores": {                                // WSZYSTKIE kategorie
    "woda_wyciek": 5.3,
    "pozar_trzask": 24.7,
    // ...
  },
  
  "all_detections": {                            // WSZYSTKIE liczniki
    "woda_wyciek": 2,
    "pozar_trzask": 12,
    // ...
  }
}
```

---

## 🔍 Diagnozy Problemów

| Problem | Objawy | Rozwiązanie |
|---------|--------|------------|
| Brak detekcji | LOG pusty | Sprawdź mikrofon (TEST), obniż MIN_CONFIDENCE |
| Zbyt wiele alarmów | 100+ detekcji w 60s | Zwiększ MIN_CONFIDENCE do 0.35 |
| Microphone error | "Błąd mikrofonu" w logu | Sprawdź audio w systemie, wybierz inne urządzenie |
| "Model nie znaleziony" | Brak ONNX | Pobierz yamnet_classifier.onnx |
| GUI zawieszony | Program nie reaguje | Zwiększ WINDOW_SEC (zbyt mały okno = CPU) |

---

## 📞 Technical Specs

| Parametr | Wartość |
|----------|---------|
| Częstotliwość próbkowania | 16 000 Hz |
| Rozmiar okna | 2.0 sekundy |
| Pokreślenie okien | 1.0 sekunda |
| Minimalna pewność | 0.25 (25%) |
| Próg ciszy | 0.008 RMS |
| Model AI | YAMNet (ONNX) |
| Latencja detekcji | ~1-2 sekund |
| Liczba kategorii | 6 |

---

## 🎓 Jak Działa?

```
Mikrofon
  ↓ (nagrywanie)
16 kHz mono audio
  ↓ (buforowanie)
Okno 2s + overlap 1s
  ↓ (ekstrakcja cech)
┌─────────────────────────┐
│ YAMNet AI (ONNX)        │  ← Jeśli model istnieje
│ Fallback Heuristics     │  ← Zawsze na backup
└─────────────────────────┘
  ↓ (klasyfikacja)
Kategoria + Confidence
  ↓ (filtrowanie)
Jeśli conf > MIN_CONFIDENCE
  ↓ (interface update)
GUI Log + Score Bars
  ↓ (agregacja)
Score Sum + Vote Count
  ↓ (na koniec misji)
JSON Report + Top3
```

---

## 🚀 Zaawansowane

### Custom Logika Klasyfikacji

Edytuj funkcję `_fallback_logic()` w pliku:
```python
def _fallback_logic(audio, f):
    # Tutaj są reguły
    # f["kurtosis"], f["rms"], f["zcr"], f["sc"]
    # Zwróć (kategoria, confidence)
```

### Live Training Fallback (v1.2+)

```python
# Przykład rekordy dla nauki (przyszłość)
TRAINING_DATA = {
    "bomba_lotnicza": [
        {"kurtosis": 15.2, "rms": 0.08, "expected": True},
        # ...
    ]
}
```

---

## 📝 License & Credits

- Model: **YAMNet** (Google)
- Framework: **ONNX Runtime**
- GUI: **Tkinter** (Python stdlib)
- Audio: **LibROSA**, **SoundDevice**

---

## 🔗 Linki

- [YAMNet Model](https://tfhub.dev/google/yamnet/1)
- [LibROSA Docs](https://librosa.org/)
- [ONNX Runtime](https://onnxruntime.ai/)

---

**Wersja:** 1.1  
**Data:** 2026-04-05  
**Status:** ✅ Production Ready  
**Autor:** DSP Advanced Team  
**Język:** Python 3.8+  
**OS:** Windows / Linux / macOS

🎯 **Powodzenia w testach!** 🚀
