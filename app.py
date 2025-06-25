# streamlit_emotion_app.py
import streamlit as st
import numpy as np
import librosa
import tempfile
import matplotlib.pyplot as plt
from keras.models import load_model

# === Load Model ===
MODEL_PATH = 'cnn+lstm_model.h5'

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# === Define emotion classes in exact order used during model training ===
emotion_classes = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# === Feature Extraction ===
def extract_mfccs(file_path):
    try:
        y, sr = librosa.load(file_path, duration=3, offset=0.5)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed
    except Exception as e:
        st.error(f"❌ Audio feature extraction failed: {e}")
        return None

# === Streamlit UI ===
st.set_page_config(page_title="Emotion Recognition from Audio")
st.title("🎙️ Speech Emotion Recognition")
st.write("Upload a `.wav` audio file and detect the emotion expressed.")

uploaded_file = st.file_uploader("Choose a WAV audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_audio_path = tmp_file.name

    mfcc = extract_mfccs(temp_audio_path)

    if mfcc is not None:
        try:
            input_data = mfcc.reshape(1, 40, 1)
            prediction = model.predict(input_data)
            predicted_index = np.argmax(prediction)
            predicted_label = emotion_classes[predicted_index]

            st.success(f"🎯 Predicted Emotion: **{predicted_label.capitalize()}**")

            # Display confidence for all classes
            st.subheader("Prediction Confidence (All Classes)")
            for i, prob in enumerate(prediction[0]):
                st.write(f"{emotion_classes[i].capitalize()}: {prob * 100:.2f}%")

            # Plot confidence bar chart
            fig, ax = plt.subplots()
            ax.barh(emotion_classes, prediction[0], color='cornflowerblue')
            ax.set_xlabel("Confidence")
            ax.set_title("Emotion Confidence Distribution")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
