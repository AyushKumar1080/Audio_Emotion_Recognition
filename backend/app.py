import logging
import tempfile
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "emotion_model.keras"
STATIC_DIR = BASE_DIR / "static"
HTML_PATH = BASE_DIR / "index.html"

STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Audio Emotion Recognition")

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static"
)

# Emotion labels used by the project
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

# Load models once when the application starts
model = tf.keras.models.load_model(str(MODEL_PATH))
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")


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
    # Load audio at 16 kHz
    audio, sr = librosa.load(path, sr=16000)

    if len(audio) == 0:
        raise ValueError("The audio file is empty.")

    # Normalize the audio
    audio = (audio - np.mean(audio)) / (np.std(audio) + 1e-8)

    # Pad or trim audio
    if len(audio) > MAX_LEN:
        audio = audio[:MAX_LEN]
    else:
        audio = np.pad(
            audio,
            (0, MAX_LEN - len(audio))
        )

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

        # Prepare features and predict emotion
        features = preprocess_audio(str(temp_path))
        predictions = model.predict(features, verbose=0)

        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))

        return {
            "emotion": emotion_labels[predicted_index],
            "confidence": round(confidence, 4)
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Audio emotion prediction failed")

        raise HTTPException(
            status_code=500,
            detail="Failed to process the audio. Check the backend logs."
        ) from exc

    finally:
        await file.close()

        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
