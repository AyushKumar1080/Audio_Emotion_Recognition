
import tempfile
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "emotion_model.keras"
YAMNET_PATH = BASE_DIR / "models" / "yamnet"

STATIC_DIR = BASE_DIR / "static"
HTML_PATH = BASE_DIR / "index.html"

app = FastAPI(title="Audio Emotion Recognition")

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static"
)

emotion_labels = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised"
]

MAX_LEN = 66200

# Load the emotion classification model
model = tf.keras.models.load_model(str(MODEL_PATH))

# Create the models directory if it does not exist
YAMNET_PATH.parent.mkdir(parents=True, exist_ok=True)

# Load YAMNet locally, or download and save it if missing
if (YAMNET_PATH / "saved_model.pb").is_file():
    yamnet = hub.load(str(YAMNET_PATH))
else:
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
    tf.saved_model.save(yamnet, str(YAMNET_PATH))

    # Reload the saved model from disk
    yamnet = hub.load(str(YAMNET_PATH))


@app.get("/", include_in_schema=False)
async def home():
    if not HTML_PATH.is_file():
        raise HTTPException(
            status_code=404,
            detail="index.html was not found in the project root."
        )

    return FileResponse(HTML_PATH)


@app.get("/health")
async def health():
    return {"status": "ok"}


def preprocess_audio(path):
    audio, sr = librosa.load(path, sr=16000)

    if len(audio) == 0:
        raise ValueError("The audio file is empty.")

    # Normalize audio
    audio = (audio - np.mean(audio)) / (np.std(audio) + 1e-8)

    # Pad or trim audio
    if len(audio) > MAX_LEN:
        audio = audio[:MAX_LEN]
    else:
        audio = np.pad(audio, (0, MAX_LEN - len(audio)))

    # Extract YAMNet embeddings
    waveform = tf.convert_to_tensor(audio, dtype=tf.float32)

    scores, embeddings, spectrogram = yamnet(waveform)

    embedding = tf.reduce_mean(embeddings, axis=0)

    return np.expand_dims(embedding.numpy(), axis=0)


@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    filename = file.filename or ""

    if Path(filename).suffix.lower() != ".wav":
        raise HTTPException(
            status_code=400,
            detail="Please upload a WAV audio file."
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)

            while chunk := await file.read(1024 * 1024):
                temp_file.write(chunk)

        if temp_path.stat().st_size == 0:
            raise HTTPException(
                status_code=400,
                detail="The uploaded audio file is empty."
            )

        # Extract features and predict
        features = preprocess_audio(str(temp_path))
        predictions = model.predict(features, verbose=0)

        predicted_index = int(np.argmax(predictions[0]))

        if predicted_index >= len(emotion_labels):
            raise ValueError(
                f"Unexpected prediction index: {predicted_index}"
            )

        confidence = float(np.max(predictions[0]))

        return {
            "emotion": emotion_labels[predicted_index],
            "confidence": round(confidence, 4)
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process audio: {type(exc).__name__}: {exc}"
        ) from exc

    finally:
        await file.close()

        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
