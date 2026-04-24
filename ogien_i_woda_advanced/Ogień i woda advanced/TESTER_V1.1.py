#!/usr/bin/env python3
"""
TESTER - Zweryfikowanie wszystkich zmian w v1.1
================================================

Uruchom: python TESTER_V1.1.py
"""

import os
import sys
import json
from datetime import datetime

# Kolory dla outputu
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
END = '\033[0m'

def test(condition, message):
    """Wypisz wynik testu"""
    symbol = f"{GREEN}✓{END}" if condition else f"{RED}✗{END}"
    print(f"  {symbol} {message}")
    return condition

def section(title):
    """Sekcja testu"""
    print(f"\n{BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{END}")
    print(f"{BLUE}  {title}{END}")
    print(f"{BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{END}\n")

def main():
    print(f"\n{YELLOW}╔════════════════════════════════════════════╗{END}")
    print(f"{YELLOW}║  TESTER - Sound Classifier v1.1             ║{END}")
    print(f"{YELLOW}║  Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}                        ║{END}")
    print(f"{YELLOW}╚════════════════════════════════════════════╝{END}")
    
    all_pass = True
    
    # TEST 1: Pliki główne istnieją
    section("TEST 1: Struktury Folderów i Plików")
    all_pass &= test(
        os.path.exists("Sound_classifier_gui2.py"),
        "Sound_classifier_gui2.py istnieje"
    )
    all_pass &= test(
        os.path.exists("yamnet_classifier.onnx"),
        "yamnet_classifier.onnx istnieje (model AI)"
    )
    
    # TEST 2: Importy mogą być załadowane (jeśli zainst.)
    section("TEST 2: Biblioteki Python")
    try:
        import tkinter
        all_pass &= test(True, "tkinter ✓")
    except ImportError:
        all_pass &= test(False, "tkinter ✗ (wbudowany, powinien być)")
    
    try:
        import numpy
        all_pass &= test(True, "numpy ✓")
    except ImportError:
        all_pass &= test(False, "numpy ✗ (zainstaluj: pip install numpy)")
    
    try:
        import librosa
        all_pass &= test(True, "librosa ✓")
    except ImportError:
        all_pass &= test(False, "librosa ✗ (zainstaluj: pip install librosa)")
    
    try:
        import sounddevice
        all_pass &= test(True, "sounddevice ✓")
    except ImportError:
        all_pass &= test(False, "sounddevice ✗ (zainstaluj: pip install sounddevice)")
    
    try:
        import onnxruntime
        all_pass &= test(True, "onnxruntime ✓")
    except ImportError:
        all_pass &= test(False, "onnxruntime ✗ (zainstaluj: pip install onnxruntime)")
    
    # TEST 3: Przykładowy raport JSON
    section("TEST 3: Format Raportu JSON")
    
    # Sprawdź czy plik PRZYKLAD_RAPORT.json istnieje
    if os.path.exists("PRZYKLAD_RAPORT.json"):
        try:
            with open("PRZYKLAD_RAPORT.json", "r") as f:
                report = json.load(f)
            
            all_pass &= test(
                "timestamp" in report,
                "Pole 'timestamp' istnieje (ISO format)"
            )
            all_pass &= test(
                "datetime_pl" in report,
                "Pole 'datetime_pl' istnieje (PL format)"
            )
            all_pass &= test(
                "duration_sec" in report,
                "Pole 'duration_sec' istnieje (nowe)"
            )
            all_pass &= test(
                "sample_rate" in report,
                "Pole 'sample_rate' istnieje (nowe)"
            )
            all_pass &= test(
                "all_detections" in report,
                "Pole 'all_detections' istnieje (nowe)"
            )
            
            # Sprawdzenie struktury top3
            all_pass &= test(
                len(report.get("top3", [])) > 0,
                "Top 3 wyniki istnieją"
            )
            
        except json.JSONDecodeError:
            all_pass &= test(False, "PRZYKLAD_RAPORT.json jest invalid")
    else:
        all_pass &= test(False, "PRZYKLAD_RAPORT.json nie znaleziony")
    
    # TEST 4: Pliki dokumentacji
    section("TEST 4: Pliki Dokumentacji i Poradników")
    all_pass &= test(
        os.path.exists("ZMIANY_I_INSTRUKCJE.md"),
        "ZMIANY_I_INSTRUKCJE.md (dokumentacja zmian)"
    )
    all_pass &= test(
        os.path.exists("PORADNIK_OPTYMALIZACJI.md"),
        "PORADNIK_OPTYMALIZACJI.md (tuning)"
    )
    all_pass &= test(
        os.path.exists("CHANGELOG.md"),
        "CHANGELOG.md (historia wersji)"
    )
    
    # TEST 5: Analiza kodu
    section("TEST 5: Analiza Kodu Sound_classifier_gui2.py")
    
    try:
        with open("Sound_classifier_gui2.py", "r") as f:
            code = f.read()
        
        all_pass &= test(
            "MIN_CONFIDENCE = 0.25" in code,
            "MIN_CONFIDENCE obniżony na 0.25 ✓"
        )
        all_pass &= test(
            "_fallback_logic" in code,
            "Funkcja _fallback_logic istnieje"
        )
        all_pass &= test(
            'if kurtosis > 10 and rms > 0.05:' in code,
            "Nowa logika dla bomby lotniczej (kurtosis > 10)"
        )
        all_pass &= test(
            'sc = float(np.mean(librosa.feature.spectral_centroid' in code 
            and '_fallback_logic' in code,
            "Spectral Centroid w fallback logice"
        )
        all_pass &= test(
            'zcr = float(np.mean(librosa.feature.zero_crossing_rate' in code,
            "Zero Crossing Rate w fallback logice"
        )
        all_pass &= test(
            'conf=conf:.2%' in code,
            "Confidence wyświetlany w procentach (conf=XX%)"
        )
        all_pass &= test(
            'rms_db = 20 * np.log10' in code,
            "RMS konwertowany na decybele w GUI"
        )
        all_pass &= test(
            'report_num = 1' in code and 'while os.path.exists' in code,
            "Numeracja rapportów zaimplementowana ✓"
        )
        all_pass &= test(
            'datetime_pl' in code,
            "Polski format daty w JSON"
        )
        all_pass &= test(
            'all_detections' in code,
            "Pole all_detections w raporcie"
        )
        
    except FileNotFoundError:
        all_pass &= test(False, "Sound_classifier_gui2.py nie znaleziony!")
    
    # TEST 6: Gotowość do uruchomienia
    section("TEST 6: Gotowość do Uruchomienia")
    
    libs_ok = True
    for lib in ["numpy", "librosa", "sounddevice", "onnxruntime"]:
        try:
            __import__(lib)
        except ImportError:
            libs_ok = False
            break
    
    all_pass &= test(
        libs_ok,
        "Wszystkie biblioteki zainstalowane" if libs_ok 
        else "Zainstaluj: pip install librosa sounddevice numpy scipy onnxruntime"
    )
    
    # WYNIK KOŃCOWY
    section("PODSUMOWANIE")
    if all_pass:
        print(f"{GREEN}✓ WSZYSTKIE TESTY PRZESZŁY!{END}")
        print(f"\n{BLUE}Możesz teraz uruchomić:{END}")
        print(f"  python Sound_classifier_gui2.py\n")
    else:
        print(f"{RED}✗ Algunóre testy nie przeszły.{END}")
        print(f"\n{YELLOW}Sprawdź wyżej które problemy się pojawiły.{END}")
        print(f"{YELLOW}Najpewniej: pip install librosa sounddevice numpy scipy onnxruntime\n{END}")
    
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())
