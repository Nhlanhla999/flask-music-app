import os
import tempfile
import subprocess
import numpy as np
import pandas as pd
import librosa
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 🔄 Config
csv_path = 'south_african_music.csv'
songs_folder = 'Music'

# 🗃️ Load dataset
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found: {csv_path}")

df = pd.read_csv(csv_path)

GENRE_TO_MOOD = {
    "House": ["relax"],
    "Afro-beat": ["romance"],
    "Hip-hop": ["energetic"],   
    "Nostalgic": ["party"], 
    "Afro-soul": ["sad"],      
    "Rap": ["feel_good"]
}

df['mood'] = df['genre'].map(GENRE_TO_MOOD)
df = df.dropna(subset=['mood'])
df['mood'] = df['mood'].apply(lambda x: x[0] if isinstance(x, list) else x)

print(f"📁 Loaded {len(df)} songs with mood labels.")

def extract_features(file_path):
    # Force standard sample rate
    y, sr = librosa.load(file_path, sr=44100, duration=30)

    # 20 MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    # 12 Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    # 6 Spectral Contrast bands (avoid Nyquist issue)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
    # 6 Tonnetz features
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)

    # Stack means
    features = np.hstack([
        np.mean(mfcc, axis=1),
        np.mean(chroma, axis=1),
        np.mean(contrast, axis=1),
        np.mean(tonnetz, axis=1)
    ])
    return features



# --- Train mood classifier using audio files only ---
X = []
y = []

AUDIO_EXTENSIONS = ('.mp3', '.wav', '.flac', '.ogg')  # mood classification only

print("🎧 Extracting features for mood classification (audio only)...")
for _, row in df.iterrows():
    filename = row['filename']
    filepath = os.path.join(songs_folder, filename)

    if not filename.lower().endswith(AUDIO_EXTENSIONS):
        print(f"⏩ Skipping {filename} (not audio)")
        continue

    if not os.path.exists(filepath):
        print(f"❌ File not found: {filename}")
        continue

    try:
        features = extract_features(filepath)
        X.append(features)
        y.append(row['mood'])
        print(f"✅ Processed: {filename}")
    except Exception as e:
        print(f"⚠️ Error processing {filename}: {e}")

if not X:
    raise RuntimeError("No audio features extracted. Check your MP3/WAV files.")

X = np.array(X)
y = np.array(y, dtype=str)

# Train mood classifier
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("\n🎯 Classification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(clf, 'mood_classifier.pkl')
print("\n✅ Mood classifier saved as 'mood_classifier.pkl'")

