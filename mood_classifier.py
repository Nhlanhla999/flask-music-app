import librosa
import os
import joblib
import numpy as np
import json


MOODS = ['feel_good', 'sad', 'energetic', 'relax','party', 'romance']

FEATURES_FILE = "features_cache.json"
MODEL_PATH = 'mood_classifier.pkl'

# Persistent cache for similarity features
audio_features_cache = {}

# --- Load model ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)


def save_features_to_disk():
    with open(FEATURES_FILE, "w") as f:
        json.dump({os.path.normpath(k): v.tolist() for k, v in audio_features_cache.items()}, f)

def load_features_cache():
    global audio_features_cache
    if os.path.exists(FEATURES_FILE):
        with open(FEATURES_FILE, "r") as f:
            audio_features_cache = {os.path.normpath(k): np.array(v) for k, v in json.load(f).items()}



def cosine_similarity(vec1, vec2):
    # Handles zero vectors
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def get_audio_features(file_path, mode="mood"):
    try:
        y, sr = librosa.load(file_path, sr=None, duration=30)

        if mode == "mood":
            # Match training
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)

            features = np.hstack([
                np.mean(mfcc, axis=1),      # 20
                np.mean(chroma, axis=1),    # 12
                np.mean(contrast, axis=1),  # 7
                np.mean(tonnetz, axis=1)    # 6
            ])  # Total: 45

        elif mode == "similarity":
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features = np.mean(mfcc, axis=1)
            audio_features_cache[file_path] = features
            save_features_to_disk()

        return features

    except Exception as e:


        print(f"❌ Error extracting features from {file_path}: {e}")
        return np.array([])

def precompute_folder_features(folder_path):
    for f in os.listdir(folder_path):
        if f.lower().endswith(".mp4"):
            fp = os.path.normpath(os.path.join(folder_path, f))
            if fp not in audio_features_cache:
                feats = get_audio_features(fp, mode='similarity')
                if feats.size > 0:
                    audio_features_cache[fp] = feats
    save_features_to_disk()



def classify_song(file_path: str) -> str:
    features = get_audio_features(file_path, mode="mood")
    if features.size != 45:   # safety check
        print(f"⚠️ Feature mismatch for {file_path}: {features.size} features")
        return "unknown"
    try:
        return model.predict([features])[0]
    except Exception as e:
        print(f"❌ Error classifying {file_path}: {e}")
        return "unknown"

