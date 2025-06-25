import numpy as np
import librosa
from keras.models import load_model

# Use direct class list if no LabelEncoder
emotion_classes = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# Load model
model = load_model("cnn+lstm_model.h5")

# Predict
file_path = "Audio_Speech_Actors_01-24/Actor_01/03-01-02-01-02-01-01.wav"
features = extract_mfcc(file_path).reshape(1, 40, 1)
prediction = model.predict(features)
predicted_label = emotion_classes[np.argmax(prediction)]

print("Predicted Emotion:", predicted_label)
