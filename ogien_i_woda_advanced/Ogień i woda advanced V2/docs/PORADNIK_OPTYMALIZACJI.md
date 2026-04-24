# Poradnik Optymalizacji Detekcji

## 🎯 Cel: Maksymalna czułość i dokładność

---

## 1️⃣ Jeśli System Nie Detektuje Dźwięków

### Prvo - Sprawdź Sprzęt
```bash
# Wylistuj dostępne urządzenia audio
python Sound_classifier_gui2.py
# Kliknij "ODŚWIEŻ" - powino widać mikrofon
# Kliknij "TEST MIKROFONU" - monitor VU meter
```

**Co powinno być widać:**
- ✅ VU meter reaguje na dźwięki
- ✅ Poziom zmienia się z "-40dB" do "-10dB" 
- ❌ Jeśli zawsze "--" → problem z mikrofonem

### Druga - Zwiększ Czułość
```python
MIN_CONFIDENCE = 0.20  # Zamiast 0.25 (więcej detekcji, więcej fałszyw)
SILENCE_RMS    = 0.005  # Zamiast 0.008 (bardziej czuły na cichsze)
```

### Trzecia - Zmień Rozmiar Okna
```python
WINDOW_SEC = 1.5  # Zamiast 2.0 (szybsze detekcje, mniej stabilne)
OVERLAP_SEC = 0.75  # Zamiast 1.0
```

---

## 2️⃣ Jeśli Jest Zbyt Wiele Fałszywych Alarmów

### Zwiększ Próg Pewności
```python
MIN_CONFIDENCE = 0.35  # Zamiast 0.25 (mniej detekcji, więcej trafnych)
```

### Zwiększ Rozmiar Okna
```python
WINDOW_SEC = 2.5  # Zamiast 2.0 (więcej stabilności, wolniej reaguje)
OVERLAP_SEC = 1.25  # Zamiast 1.0
```

---

## 3️⃣ Strategie Testowania

### Test 1: Kalibracja na Szumie Tła
1. Uruchom aplikację
2. Nie wydawaj żadnych dźwięków przez 30 sekund
3. Obserwuj czarny tekst w logu - nie powinno być detekcji
4. **Jeśli jest** → zbyt niska wartość `SILENCE_RMS`

```python
# Reguła: jeśli fałszywe alarmy, to zwiększ o 0.002
SILENCE_RMS = 0.010  # Zamiast 0.008
```

### Test 2: Czułość Mikrofonu
1. **Ciche dźwięki** - szeptem coś powiedz
   - Powino być: `-30dB` do `-20dB` w VU metrze
   - Powino być: co najmniej kilka detekcji
   
2. **Normalny dźwięk** - mów głośno
   - Powino być: `-15dB` do `-5dB`
   - Powino być: detekcje co sekundę
   
3. **Głośny dźwięk** - krzyknij
   - Powino być: `0dB` do `-5dB`
   - Powino być: ciągłe detekcje

### Test 3: Dokładność Klasyfikacji
Nagram oddzielnie każdy typ dźwięku:
```
bomba_lotnicza      → Nagraj eksplozję / skok (impuls)
serie_bron          → Nagraj szybkie kliknięcia
pozar_trzask        → Nagraj trzask papieru/drewna
woda_wyciek         → Nagraj szmer wody
kolumna_pancerna    → Nagraj silnik samochodu (niska częst.)
```

---

## 4️⃣ Zaawansowane Ustawienia

### Parametry Klasyfikacji (w funkcji `_fallback_logic`)

```python
# Bomba lotnicza - duży kurtosis (impulsowy)
if kurtosis > 10 and rms > 0.05:
    return "bomba_lotnicza", min(0.9, 0.5 + kurtosis * 0.04)

# ZMIANA: jeśli zbyt czuły na bomby
if kurtosis > 15 and rms > 0.06:  # Wyższe progi
    return "bomba_lotnicza", min(0.85, 0.5 + kurtosis * 0.03)
```

### Zmiana Wagi Cech

```python
# Domyślnie: wszystkie cechy mają równe znaczenie
# Aby zwiększyć znaczenie RMS:
if rms > 0.05:  # Większy wkład energii
    confidence *= 1.2  # Boost pewności
```

---

## 5️⃣ Monitorowanie Wydajności

### Metrika: Wskaźnik Detekcji
```
Ideal: 2-5 detekcji na sekundę konkretnego dźwięku
Niska: <1 detekcja na sekundę
Wysoka: >10 detekcji na sekundę (fałszywe alarmy?)
```

### Metrika: Pewność (confidence)
```
Spora pewność (>0.7): system ma duże zaufanie
Średnia pewność (0.4-0.7): system jest niepewny
Niska pewność (<0.4): zazwyczaj fałszywy alarm
```

### Analiza Raportu JSON
```json
{
  "all_detections": {
    "bomba_lotnicza": 0,  // ← Brak detekcji? Problem!
    "serie_bron": 8       // ← Normalne
  },
  "all_scores": {
    "bomba_lotnicza": 0.0,  // ← Powinno być >0 dla rzeczywistych
    "serie_bron": 6.5       // ← OK
  }
}
```

---

## 6️⃣ Szybki Checklist przed Testami Polowymi

- [ ] Mikrofon pracuje (VU meter reaguje)
- [ ] "MODUŁ AI (ONNX) ZAŁADOWANY" widoczny w logu
- [ ] Test na wiadomych dźwiękach przechodzi
- [ ] MIN_CONFIDENCE dostrojony do obszaru
- [ ] Fałszywe alarmy akceptowalne (<5% w ciszy)
- [ ] Folder roboczy ma prawa do zapisuJSON
- [ ] Deckcie z logu pokrywają się z rzeczywistością

---

## 7️⃣ Debugowanie Logów

### Normalny Log Startu
```
[14:32:15] ==================== info
[14:32:15] MODUŁ AI (ONNX) ZAŁADOWANY POMYŚLNIE. info
[14:32:15] Znaleziono 2 urządzenia. info
[14:32:15] ==================== info
```

### Log Detekcji
```
[14:32:20] ~ Woda / wyciek        [conf=67%] RMS=−15.2dB ...  cat
[14:32:21] # Pozar / trzask       [conf=58%] RMS=−12.1dB ...  cat
[14:32:22] = Kolumna pancerna     [conf=45%] RMS=−18.0dB ...  warn
```

### Problem: Zbyt Dużo Warnów
```
[14:32:23] ~ Woda / wyciek        [conf=22%] RMS=−25.0dB ...  warn ← Niski confidence
```
→ Zwiększ `MIN_CONFIDENCE` lub `SILENCE_RMS`

---

## 📞 Support

Jeśli nic nie działa:
1. Sprawdź czy yamnet_classifier.onnx jest w folderze
2. Uruchom: `python -c "import librosa; print(librosa.__version__)"`
3. Sprawdź logi z ostatniego okresu
4. Prześlij cały output logu

Powodzenia! 🚀
