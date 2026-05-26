from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import numpy as np
import uvicorn
import tempfile
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "emotion_model.keras"
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Load YAMNet
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

# Same labels used in training
emotion_labels=["neutral","calm","happy","sad","angry",
    "fearful","disgust","surprised"]

MAX_LEN=66200

def preprocess_audio(path):

    y,sr=librosa.load(path,sr=16000)

    y=(y-np.mean(y))/(np.std(y)+1e-8)

    if len(y)>MAX_LEN:
        y=y[:MAX_LEN]
    else:
        y=np.pad(y,(0,MAX_LEN-len(y)))

    waveform=tf.convert_to_tensor(y,dtype=tf.float32)

    scores,embeddings,spectrogram=yamnet(waveform)

    embedding=tf.reduce_mean(embeddings,axis=0)

    return np.expand_dims(embedding.numpy(),axis=0)


@app.post('/predict')
async def predict(file:UploadFile=File(...)):

    with tempfile.NamedTemporaryFile(delete=False,suffix='.wav') as tmp:

        content=await file.read()
        tmp.write(content)
        temp_path=tmp.name

    x=preprocess_audio(temp_path)

    prediction=model.predict(x)

    pred_class=np.argmax(prediction)

    confidence=float(np.max(prediction))

    return {
        "emotion":emotion_labels[pred_class],
        "confidence":round(confidence,4)
    }


if __name__=='__main__':
    uvicorn.run(app,host='0.0.0.0',port=8000)
