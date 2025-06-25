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

```python
     def extract_mfccs(filename):
     y, sr = librosa.load(filename, duration=3, offset=0.5)
     mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
     return mfccs

    features = []
    label_values = []

    for _, row in df_combined.iterrows():
        path = row['path']
        label = row['labels']
        mfcc = extract_mfccs(path)

        if mfcc is not None:
            features.append(mfcc)
            label_values.append(label)
```

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

```python
    model = Sequential([
    Conv1D(128, kernel_size=3, activation='relu', padding='same', input_shape=(40, 1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    Conv1D(256, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    Conv1D(256, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    LSTM(128, return_sequences=True),
    LSTM(256, return_sequences=True),
    Dropout(0.3),
    LSTM(64),

    Dense(len(label_encoder.classes_), activation='softmax')
])
```
