"""
TRENING MODELU AI — Droniada 2026
===================================
Nie wymaga tensorflow! Działa na Python 3.13.

Wymagania:
    pip install librosa numpy scikit-learn

Uruchomienie:
    python trenuj_model.py

Po zakończeniu powstanie plik:
    C:\\test\\ProgramV2\\model_gotowy.pkl
"""

import os
import glob
import numpy as np
import librosa
import pickle

try:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
except ImportError:
    print("=" * 55)
    print("BLAD: Brak scikit-learn!")
    print("Uruchom:")
    print("   pip install scikit-learn librosa numpy")
    print("=" * 55)
    input("Nacisnij Enter zeby zamknac...")
    exit(1)


# ══════════════════════════════════════════════════════════════
# USTAWIENIA
# ══════════════════════════════════════════════════════════════

FOLDER_PROBEK = r"C:\test\ProgramV2\probki"
FOLDER_ZAPISU = r"C:\test\ProgramV2"
SAMPLE_RATE   = 22050
OKNO_SEKUND   = 2.0
PROG_CISZY    = 0.005

OKNO_PROBEK = int(SAMPLE_RATE * OKNO_SEKUND)
KROK_PROBEK = OKNO_PROBEK // 2

KLASY = [
    "bomba_lotnicza",
    "kolumna_pancerna",
    "pozar_trzask",
    "serie_broni_maszynowej",
    "woda_wyciek",
    "wystrzal_krab",
]

NAZWY_PL = {
    "bomba_lotnicza":         "Bomba lotnicza",
    "kolumna_pancerna":       "Kolumna pancerna",
    "pozar_trzask":           "Pozar (trzask)",
    "serie_broni_maszynowej": "Serie z broni maszynowej",
    "woda_wyciek":            "Woda (wyciek)",
    "wystrzal_krab":          "Wystrzal Krab",
}


# ══════════════════════════════════════════════════════════════
# EKSTRAKCJA CECH
# ══════════════════════════════════════════════════════════════

def wyciagnij_cechy(audio, sr):
    """
    Zamienia surowe audio na wektor liczb opisujacych dzwiek.

    MFCC — opisuje "barwe" dzwieku (najwazniejsza cecha)
    Centroid widmowy — "srodek ciezkosci" czestotliwosci
    ZCR — jak czesto sygnal przecina zero
    RMS — glosnosc
    Energia niskopasmowa — ile energii w basach
    Kurtosis — impulsowos sygnalu (wysokie = strzaly)
    """
    cechy = []

    # MFCC: 13 wspolczynnikow x (srednia + odchylenie + delta) = 39 liczb
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    cechy.extend(np.mean(mfcc, axis=1))
    cechy.extend(np.std(mfcc,  axis=1))
    delta = librosa.feature.delta(mfcc)
    cechy.extend(np.mean(delta, axis=1))

    # Centroid widmowy
    sc = librosa.feature.spectral_centroid(y=audio, sr=sr)
    cechy.append(float(np.mean(sc)))
    cechy.append(float(np.std(sc)))

    # Rolloff widmowy
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)
    cechy.append(float(np.mean(rolloff)))
    cechy.append(float(np.std(rolloff)))

    # Szerokosc pasma widmowego
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    cechy.append(float(np.mean(bandwidth)))
    cechy.append(float(np.std(bandwidth)))

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(audio)
    cechy.append(float(np.mean(zcr)))
    cechy.append(float(np.std(zcr)))

    # RMS (glosnosc)
    rms = librosa.feature.rms(y=audio)
    cechy.append(float(np.mean(rms)))
    cechy.append(float(np.std(rms)))

    # Kontrast widmowy
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    cechy.extend(np.mean(contrast, axis=1))

    # Energia niskopasmowa (< 500 Hz)
    stft     = np.abs(librosa.stft(audio))
    freqs    = librosa.fft_frequencies(sr=sr)
    low_e    = float(stft[freqs < 500].sum() / (stft.sum() + 1e-9))
    cechy.append(low_e)

    # Kurtosis (impulsowos)
    mu   = np.mean(audio)
    sig  = np.std(audio) + 1e-9
    kurt = float(np.mean(((audio - mu) / sig) ** 4))
    cechy.append(min(kurt, 100.0))

    return np.array(cechy, dtype=np.float32)


# ══════════════════════════════════════════════════════════════
# ZBIERANIE DANYCH
# ══════════════════════════════════════════════════════════════

def zbierz_dane():
    print()
    print("=" * 60)
    print("  KROK 1: Wczytywanie i analiza probek dzwiekowych")
    print("=" * 60)

    wszystkie_cechy    = []
    wszystkie_etykiety = []
    podsumowanie       = {}

    for numer_klasy, nazwa_klasy in enumerate(KLASY):
        folder_klasy = os.path.join(FOLDER_PROBEK, nazwa_klasy)

        if not os.path.isdir(folder_klasy):
            print(f"\n  [!] Brak folderu: {folder_klasy}")
            continue

        pliki = (
            glob.glob(os.path.join(folder_klasy, "*.wav"))  +
            glob.glob(os.path.join(folder_klasy, "*.mp3"))  +
            glob.glob(os.path.join(folder_klasy, "*.flac")) +
            glob.glob(os.path.join(folder_klasy, "*.ogg"))
        )

        if not pliki:
            print(f"\n  [!] Brak plikow audio w: {folder_klasy}")
            continue

        nazwa_pl = NAZWY_PL.get(nazwa_klasy, nazwa_klasy)
        print(f"\n  [{nazwa_pl}]  -  {len(pliki)} plikow")

        okien_z_klasy = 0

        for sciezka in pliki:
            nazwa_pliku = os.path.basename(sciezka)
            try:
                # Wczytaj — librosa automatycznie konwertuje format i Hz
                audio, _ = librosa.load(sciezka, sr=SAMPLE_RATE, mono=True)

                okien_z_pliku = 0

                for start in range(0, len(audio) - OKNO_PROBEK, KROK_PROBEK):
                    okno = audio[start : start + OKNO_PROBEK]

                    # Pomin cisze
                    if float(np.sqrt(np.mean(okno ** 2))) < PROG_CISZY:
                        continue

                    cechy = wyciagnij_cechy(okno, SAMPLE_RATE)
                    wszystkie_cechy.append(cechy)
                    wszystkie_etykiety.append(numer_klasy)
                    okien_z_pliku += 1

                okien_z_klasy += okien_z_pliku
                print(f"    OK  {nazwa_pliku:<45} {okien_z_pliku} okien")

            except Exception as blad:
                print(f"    BLAD  {nazwa_pliku:<43} {blad}")

        podsumowanie[nazwa_klasy] = okien_z_klasy
        print(f"    Lacznie z klasy: {okien_z_klasy} okien")

    print()
    print("  Zebrane dane:")
    lacznie = 0
    for klasa in KLASY:
        ile = podsumowanie.get(klasa, 0)
        lacznie += ile
        print(f"    {NAZWY_PL.get(klasa, klasa):<35} {ile:>4} okien")
    print(f"    {'LACZNIE':<35} {lacznie:>4} okien")

    if lacznie == 0:
        print()
        print("  BLAD: Nie znaleziono zadnych probek!")
        input("Nacisnij Enter zeby zamknac...")
        exit(1)

    return (np.array(wszystkie_cechy,    dtype=np.float32),
            np.array(wszystkie_etykiety, dtype=np.int32))


# ══════════════════════════════════════════════════════════════
# TRENING SVM
# ══════════════════════════════════════════════════════════════

def trenuj_svm(X, y):
    """
    SVM szuka plaszczyzn ktore najlepiej oddzielaja klasy
    z jak najwiekszym marginesem bezpieczenstwa.
    Swietnie dziala przy malej ilosci probek (15-30 na klase).

    kernel='rbf'     — nieliniowe granice miedzy klasami
    C=10             — jak mocno karac bledy
    probability=True — zwraca prawdopodobienstawa (0-100%)
    """
    print()
    print("=" * 60)
    print("  KROK 2: Trening modelu SVM")
    print("=" * 60)
    print(f"  Probki: {len(X)}  |  Cechy: {X.shape[1]}  |  Klasy: {len(KLASY)}")
    print()

    # Normalizacja — KONIECZNA dla SVM
    print("  Normalizuje cechy...")
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    # Walidacja krzyzowa — uczciwa ocena jak model radzi sobie z nowymi danymi
    # Dzieli dane na 5 czesci, trenuje na 4, testuje na 1, powtarza 5 razy
    n_splits = min(5, min(np.bincount(y)))  # nie wiecej foldow niz najmniejsza klasa
    n_splits = max(2, n_splits)

    print(f"  Walidacja krzyzowa ({n_splits}-fold) — chwile poczekaj...")

    svm = SVC(kernel="rbf", C=10, gamma="scale",
              probability=True, random_state=42)

    wyniki_cv = cross_val_score(svm, X_norm, y,
                                cv=n_splits, scoring="accuracy")

    print()
    print(f"  Wyniki walidacji krzyzowej:")
    for i, wynik in enumerate(wyniki_cv, 1):
        pasek = "=" * int(wynik * 30) + "-" * (30 - int(wynik * 30))
        print(f"    Fold {i}: {wynik*100:5.1f}%  [{pasek}]")

    srednia = wyniki_cv.mean()
    print(f"\n    Srednia: {srednia*100:.1f}%  (+/-{wyniki_cv.std()*100:.1f}%)")
    print()

    if srednia >= 0.85:
        print("  >> Swietny wynik! Model bedzie dobrze rozpoznawal dzwieki.")
    elif srednia >= 0.70:
        print("  >> Dobry wynik. Wiecej probek poprawi dokladnosc.")
    elif srednia >= 0.50:
        print("  >> Sredni wynik. Dodaj wiecej probek (cel: 30+ na klase).")
    else:
        print("  >> Niska dokladnosc. Sprawdz czy probki sa w dobrych folderach.")

    # Trenuj finalny model na WSZYSTKICH danych
    print()
    print("  Trenuję finalny model na wszystkich danych...")
    svm.fit(X_norm, y)
    print("  OK Model wytrenowany!")

    return svm, scaler


# ══════════════════════════════════════════════════════════════
# WALIDACJA SZCZEGOLOWA
# ══════════════════════════════════════════════════════════════

def waliduj(svm, scaler, X, y):
    print()
    print("=" * 60)
    print("  KROK 3: Dokladnosc per klasa")
    print("=" * 60)

    X_norm = scaler.transform(X)
    y_pred = svm.predict(X_norm)

    print()
    print(f"  {'Klasa':<30} {'Poprawne':>10} {'Lacznie':>10} {'Dok.':>8}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*8}")

    for i, klasa in enumerate(KLASY):
        maska    = y == i
        lacznie  = int(maska.sum())
        if lacznie == 0:
            continue
        poprawne = int(np.sum(y_pred[maska] == i))
        dok      = 100.0 * poprawne / lacznie
        ikona    = "OK" if dok >= 70 else ("~~" if dok >= 50 else "!!")
        print(f"  {ikona} {NAZWY_PL.get(klasa, klasa):<28} "
              f"{poprawne:>10} {lacznie:>10} {dok:>7.1f}%")

    og = 100.0 * int(np.sum(y_pred == y)) / len(y)
    print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*8}")
    print(f"  {'LACZNIE':<30} {int(np.sum(y_pred==y)):>10} "
          f"{len(y):>10} {og:>7.1f}%")


# ══════════════════════════════════════════════════════════════
# ZAPIS MODELU
# ══════════════════════════════════════════════════════════════

def zapisz_model(svm, scaler):
    print()
    print("=" * 60)
    print("  KROK 4: Zapisywanie modelu")
    print("=" * 60)

    pakiet = {
        "svm":      svm,
        "scaler":   scaler,
        "klasy":    KLASY,
        "nazwy_pl": NAZWY_PL,
        "sr":       SAMPLE_RATE,
        "okno":     OKNO_SEKUND,
    }

    sciezka = os.path.join(FOLDER_ZAPISU, "model_gotowy.pkl")
    with open(sciezka, "wb") as f:
        pickle.dump(pakiet, f)

    rozmiar = os.path.getsize(sciezka) / 1024
    print(f"  OK Model zapisany: {sciezka}")
    print(f"     Rozmiar: {rozmiar:.1f} KB")
    return sciezka


# ══════════════════════════════════════════════════════════════
# GLOWNA FUNKCJA
# ══════════════════════════════════════════════════════════════

def main():
    print()
    print("=" * 62)
    print("   TRENING MODELU AI - DRONIADA 2026")
    print("   Algorytm: SVM + MFCC  (dziala bez tensorflow!)")
    print("=" * 62)
    print()
    print(f"  Folder probek: {FOLDER_PROBEK}")
    print(f"  Folder zapisu: {FOLDER_ZAPISU}")
    print()
    print("  Klasy i liczba plikow:")

    for klasa in KLASY:
        folder = os.path.join(FOLDER_PROBEK, klasa)
        if os.path.isdir(folder):
            pliki = (glob.glob(os.path.join(folder, "*.wav")) +
                     glob.glob(os.path.join(folder, "*.mp3")) +
                     glob.glob(os.path.join(folder, "*.flac")))
            n      = len(pliki)
            status = f"{n} plikow"
            ikona  = "OK" if n >= 10 else ("~~" if n >= 5 else "!!")
        else:
            status = "BRAK FOLDERU"
            ikona  = "!!"
        print(f"    {ikona}  {NAZWY_PL.get(klasa, klasa):<35} {status}")

    print()
    input("  Nacisnij Enter zeby rozpoczac...")

    X, y        = zbierz_dane()
    svm, scaler = trenuj_svm(X, y)
    waliduj(svm, scaler, X, y)
    zapisz_model(svm, scaler)

    print()
    print("=" * 62)
    print("   GOTOWE!")
    print("   Plik model_gotowy.pkl zapisany w C:\\test\\ProgramV2\\")
    print("   Mozesz teraz uzyc go w programie do nasluchu.")
    print("=" * 62)
    print()
    input("  Nacisnij Enter zeby zamknac...")


if __name__ == "__main__":
    main()
