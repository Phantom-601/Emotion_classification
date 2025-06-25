# 🎙️ Emotion Recognition from Audio using Deep Learning

This project detects emotions from audio clips (both **speech** and **song**) using the **RAVDESS** dataset. It includes:

-   A complete **data pipeline**
-   A **hybrid deep learning model** (Conv1D + LSTM)
-   A trained model saved in `.h5` format
-   A fully functional **Streamlit web app**
-   A testable Python script
-   Requirements and documentation for reproducibility

---

## 🧠 Project Description

This project builds a deep learning model to classify emotions in audio files using the **RAVDESS** dataset. Emotions detected:

-   Neutral
-   Calm
-   Happy
-   Sad
-   Angry
-   Fearful
-   Disgust
-   Surprised

The final model is deployed as a **Streamlit web app** where users can upload a `.wav` file and get the predicted emotion in real-time.

---

## ⚙️ Preprocessing Methodology

-   Loaded audio clips from both speech and song folders
-   Extracted emotion labels from filenames
-   Used `librosa` to extract **MFCCs** (40 features) per file
-   Applied **RandomOverSampler** to balance training data
-   Encoded labels using `LabelEncoder` and `to_categorical`
-   Reshaped MFCCs into 3D shape suitable for LSTM input

---

## 🧩 Model Pipeline

We used a **hybrid CNN + LSTM architecture**:

```text
Conv1D(128) → BatchNorm → MaxPool → Dropout
Conv1D(256) → BatchNorm → MaxPool → Dropout
Conv1D(256) → BatchNorm → MaxPool → Dropout
LSTM(128) → LSTM(256) → Dropout → LSTM(64)
Dense(8) with softmax


```
