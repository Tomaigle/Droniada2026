import os
import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
import pickle

print("Ładowanie modelu YAMNet...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

def load_wav_16k_mono(filename):
    wav, sample_rate = librosa.load(filename, sr=16000, mono=True)
    return wav

def extract_embeddings(wav_data):
    scores, embeddings, spectrogram = yamnet_model(wav_data)
    return np.mean(embeddings.numpy(), axis=0)

data_dir = "dataset"
X, y = [], []

print("Przetwarzanie próbek audio (to może chwilę potrwać)...")
for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)
    if os.path.isdir(class_dir):
        for file in os.listdir(class_dir):
            if file.endswith('.wav'):
                wav_data = load_wav_16k_mono(os.path.join(class_dir, file))
                X.append(extract_embeddings(wav_data))
                y.append(class_name)

X, y = np.array(X), np.array(y)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Trenowanie własnego modelu...")
my_model = models.Sequential([
    layers.Input(shape=(1024,)),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(label_encoder.classes_), activation='softmax')
])

my_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
my_model.fit(X, y_encoded, epochs=30, batch_size=4)

# ZAPISYWANIE MODELU I ETYKIET DO PLIKÓW
my_model.save('moj_model.keras')
with open('klasy.pkl', 'wb') as f:
    pickle.dump(label_encoder.classes_, f)

print("Gotowe! Model ('moj_model.keras') i klasy zapisanane pomyślnie.")
