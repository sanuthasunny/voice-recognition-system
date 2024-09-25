pip install librosa
import os
import librosa
import numpy as np

# Load the audio file and extract features (Mel-spectrogram)
def load_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to log scale
    return mel_spec_db.T  # Transpose to fit the (time, feature) shape

# Example path to an audio file in UrbanSound8K
audio_file_path = "UrbanSound8K/fold1/101415-3-0-2.wav"
features = load_audio_features(audio_file_path)
print(features.shape)  # Should return (time_steps, n_mels)

# You can reshape it to fit into Conv1D, e.g., (time_steps, n_mels, 1)
features = np.expand_dims(features, axis=-1)
Sure! Here’s a basic example of how you might start implementing a voice modulation detection system using Python. This example will cover the following components:

1. *Feature Extraction*: Using librosa to extract features from audio.
2. *Model Training*: Training a simple classifier using scikit-learn.
3. *Prediction*: Using the trained model to make predictions on new audio data.

### Prerequisites

You need to install the following Python libraries:

bash
pip install librosa scikit-learn numpy


### 1. *Feature Extraction*


python
import librosa
import numpy as np

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    
    # Extract pitch (fundamental frequency)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = np.mean([np.mean(pitches[magnitudes > np.median(magnitudes)]) for i in range(pitches.shape[1])])
    
    # Extract tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    # Extract volume (RMS)
    volume = np.mean(librosa.feature.rms(y=y))
    
    return np.array([pitch, tempo, volume])

# Example usage
audio_path = 'example_audio.wav'
features = extract_features(audio_path)
print("Extracted features:", features)


### 2. *Model Training*


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Dummy dataset: [pitch, tempo, volume] and labels
X = np.array([
    [100, 120, 0.02],  # Example features
    [150, 130, 0.03],
    [80, 110, 0.01],
    [160, 140, 0.05]
])
y = np.array([0, 0, 1, 1])  # Labels: 0 = 'calm', 1 = 'excited'

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create and train the classifier
clf = RandomForestClassifier(n_estimators=10)
clf.fit(X_train, y_train)

# Evaluate the classifier
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))



def predict_situation(audio_path, model):
    features = extract_features(audio_path).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]

# Example usage
model = clf  # Use the trained model from above
new_audio_path = 'new_audio.wav'
situation = predict_situation(new_audio_path, model)
print("Predicted situation:", 'excited' if situation == 1 else 'calm')


