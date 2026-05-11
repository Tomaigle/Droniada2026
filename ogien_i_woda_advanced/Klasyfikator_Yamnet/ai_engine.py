import os
import threading
import queue
import numpy as np
import pickle
import librosa
from config import *

# Blokujemy irytujące logi z TensorFlow w konsoli
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import tensorflow_hub as hub

def zaladuj_model():
    if not os.path.exists(MODEL_KERAS_SCIEZKA) or not os.path.exists(KLASY_PKL_SCIEZKA):
        return None
    try:
        print("Trwa ładowanie modelu YAMNet (to potrwa chwilkę)...")
        yamnet = hub.load('https://tfhub.dev/google/yamnet/1')
        print("Trwa ładowanie Twojego klasyfikatora...")
        my_model = tf.keras.models.load_model(MODEL_KERAS_SCIEZKA)
        
        with open(KLASY_PKL_SCIEZKA, "rb") as f:
            klasy = pickle.load(f)
            
        return {"yamnet": yamnet, "my_model": my_model, "klasy": klasy}
    except Exception as e:
        print(f"Błąd ładowania modeli AI: {e}")
        return None

class KlasyfikatorWatek:
    def __init__(self, model_pakiet, wynik_queue):
        self.yamnet = model_pakiet["yamnet"]
        self.my_model = model_pakiet["my_model"]
        self.klasy = model_pakiet["klasy"]
        self.sr_model = 16000

        self.wynik_q  = wynik_queue
        self.audio_q  = queue.Queue(maxsize=5)
        self.running  = False
        self._thread  = None

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._petla, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def dodaj_audio(self, audio_chunk):
        try:
            self.audio_q.put_nowait(audio_chunk)
        except queue.Full: pass

    def _petla(self):
        bufor = []
        potrzebne = int(self.sr_model * KLASYFIKUJ_CO_SEK)

        while self.running:
            try:
                chunk = self.audio_q.get(timeout=0.5)
                bufor.append(chunk)
                lacznie = sum(len(c) for c in bufor)

                if lacznie >= potrzebne:
                    audio = np.concatenate(bufor)

                    if SAMPLE_RATE != self.sr_model:
                        audio = librosa.resample(audio, orig_sr=SAMPLE_RATE, target_sr=self.sr_model)

                    okno = int(self.sr_model * KLASYFIKUJ_CO_SEK)
                    if len(audio) > okno:
                        start = (len(audio) - okno) // 2
                        audio = audio[start : start + okno]

                    rms = float(np.sqrt(np.mean(audio ** 2)))
                    if rms < 0.01: 
                        bufor = []
                        try: self.wynik_q.put_nowait({"klasa": "—", "pewnosc": 0.0, "wszystkie": {}, "cisza": True})
                        except queue.Full: pass
                        continue

                    scores, embeddings, spectrogram = self.yamnet(audio)
                    mean_embedding = np.mean(embeddings.numpy(), axis=0)

                    prediction = self.my_model.predict(np.expand_dims(mean_embedding, axis=0), verbose=0)[0]

                    best_idx  = int(np.argmax(prediction))
                    best_klas = self.klasy[best_idx]
                    best_conf = float(prediction[best_idx])
                    wszystkie = {self.klasy[i]: float(prediction[i]) for i in range(len(self.klasy))}

                    if best_klas == "szumy" or best_conf < 0.80:
                        bufor = []
                        try:
                            self.wynik_q.put_nowait({"klasa": "—", "pewnosc": 0.0, "wszystkie": {}, "cisza": True})
                        except queue.Full: pass
                        continue

                    try:
                        self.wynik_q.put_nowait({
                            "klasa": best_klas,
                            "pewnosc": best_conf,
                            "wszystkie": wszystkie,
                            "cisza": False,
                        })
                    except queue.Full: pass

                    bufor = []

            except queue.Empty: continue
            except Exception as e: print(f"Błąd klasyfikacji: {e}"); bufor = []
