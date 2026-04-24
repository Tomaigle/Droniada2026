# Poprawki Klasyfikatora Dźwięków Bojowych v1.1

## 📋 Podsumowanie zmian

Kod został ulepszony w trzech głównych obszarach:

### 1. ✅ Lepsza detekcja dźwięków
- **Ulepszona logika fallback'u** - zamiast prostych reguł, teraz używa:
  - Kurtosis (uderzeniowość) - detektuje eksplozje
  - RMS (energia sygnału) - kontroluje czułość
  - Spectral Centroid (centrum spektralne) - rozróżnia częstotliwości
  - Zero Crossing Rate - detektuje szybkie przejścia w sygnale
  
- **Min confidence obniżona z 0.35 na 0.25** - bardziej czuła na słabe dźwięki

- **Hybrid mode** - jeśli AI zwróci niską pewność (<0.3), system spróbuje logiki heurystycznej

### 2. 🎨 Lepsze wyświetlanie w GUI
**Detekcje teraz pokazują więcej szczegółów:**
```
[14:32:15] ~ Woda / wyciek        [conf=67%] RMS=−15.2dB  Kurt=2.3  ZCR=0.045  SC=1200Hz
#  Pozar / trzask               [conf=58%] RMS=−12.1dB  Kurt=4.1  ZCR=0.067  SC=2456Hz
```

**Informacje wyświetlane:**
- 🎯 Ikona i nazwa kategorii
- 📊 Confidence (pewność) w procentach
- 🔊 RMS w decybelach (poziom głośności)
- 📈 Kurtosis (mara "ostości" dźwięku)
- 🌊 ZCR (szybkość zmian amplitudy)
- 🎼 SC - gdzie w spektrum jest dźwięk

### 3. 📁 Poprawiony zapis raportów JSON
**Nowy format nazwy:** `report_YYYYMMDD_HHMMSS_NNN.json`
- Data i godzina w formacie normalnym
- **NNN** - numeracja (001, 002, 003...) dla tego samego czasu
- Automatycznie zwiększa się numerator, jeśli raport istnieje

**Rozszerzony zawartość raportu:**
```json
{
  "timestamp": "2026-04-05T14:32:15.123456",
  "datetime_pl": "2026-04-05 14:32:15",
  "duration_sec": 360,
  "sample_rate": 16000,
  "top3": [
    {
      "rank": 1,
      "category": "bomba_lotnicza",
      "label": "Bomba lotnicza",
      "type": "przerywany",
      "score": 52.5,
      "detections": 28
    }
  ],
  "all_scores": { ... },
  "all_detections": { ... }
}
```

## 🚀 Jak uruchamiać

```bash
# Podstawowe uruchomienie
python Sound_classifier_gui2.py

# Wymagane biblioteki (jeśli jeszcze nie zainstalowane)
pip install librosa sounddevice numpy scipy onnxruntime tkinter
```

## ⚙️ Parametry do dostrojenia

W pliku edytuj sekcję `PARAMETRY`:

```python
SAMPLE_RATE    = 16000   # Częstotliwość próbkowania [Hz]
WINDOW_SEC     = 2.0     # Rozmiar okna analizy [sekund]
OVERLAP_SEC    = 1.0     # Pokreślenie okien [sekund]
SILENCE_RMS    = 0.008   # Próg ciszy [siła sygnału]
MIN_CONFIDENCE = 0.25    # Minimalny poziom pewności [0.0-1.0]
```

### Rekomendacje:
- Jeśli **zbyt wiele fałszywych alarmów** → zwiększ `MIN_CONFIDENCE` do 0.35
- Jeśli **zbyt mało detekcji** → obniż `MIN_CONFIDENCE` do 0.15-0.20
- Jeśli **za szybko** → zwiększ `WINDOW_SEC` do 3.0
- Jeśli **za wolno** → zmniejsz `WINDOW_SEC` do 1.5

## 📊 Cechy wyodrębniane przez model

1. **Kurtooza** (kurtosis) - detektuje impulsy/uderzenia
2. **RMS Energy** - ogólny poziom głośności
3. **Zero Crossing Rate** - szybkość zmian amplitudy
4. **Spectral Centroid** - gdzie w spektrum (Hz) jest energia
5. **Spectral Flatness** - gładkość spektru

## 🧪 Testowanie

### Test mikrofonu
1. Kliknij **"TEST MIKROFONU"** w GUI
2. Powinien pokazywać VU meter z poziomu mikrofonutu
3. Kliknij ponownie aby zatrzymać

### Test na pliku WAV (opcja przyszłościowa)
```bash
python Sound_classifier.py --file próbka.wav
```

## 🐛 Troubleshooting

| Problem | Rozwiązanie |
|---------|------------|
| Import error librosa | `pip install librosa` |
| Import error sounddevice | `pip install sounddevice` |
| "Brak urządzeń audio" | Sprawdź audio w systemie |
| Model .onnx nie znaleziony | Upewnij się że `yamnet_classifier.onnx` jest w folderze |
| Nie ma detekcji | Zwiększ poziom głośności mikrofonu lub obniż MIN_CONFIDENCE |

## 📝 Struktura kategorii

| Kategoria | Typ | Opis |
|-----------|-----|------|
| woda_wyciek | Ciągły | Hałas rozlewającym się cieczy |
| pozar_trzask | Ciągły | Trzask drewna/pożaru |
| wystrzal_krab | Przerywany | Pojedynczy strzał |
| bomba_lotnicza | Przerywany | Eksplozja lotnicza |
| kolumna_pancerna | Ciągły | Silnik czołgu |
| serie_bron_maszynowej | Przerywany | Seria strza |

## 📈 Wynik końcowy

Po zatrzymaniu misji widać:
1. **Wynik misji — TOP 3** w logu
2. **Raport JSON** zapisany w folderze
3. **Wykresy** w panelu wyników z score dla każdej kategorii

---

✨ **Wersja:** 1.1  
🔧 **Ostatnia aktualizacja:** 2026-04-05  
💻 **Python:** 3.8+  
