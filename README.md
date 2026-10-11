# Audio Emotion Recognition

An audio-based emotion recognition application that predicts the emotion
expressed in a speech recording. The project uses **YAMNet** to extract
audio embeddings and a trained **TensorFlow/Keras** classification model
to predict one of eight emotion classes. A **FastAPI** backend serves
the prediction API, while an HTML, CSS, and JavaScript frontend lets
users upload WAV files and view the result.

## Features

-   Upload speech recordings in `.wav` format.
-   Load and preprocess audio with Librosa at a 16 kHz sample rate.
-   Normalize audio and pad or trim it to the configured input length.
-   Extract embeddings with YAMNet and average them across audio frames.
-   Predict one of eight emotion labels:
    -   Neutral
    -   Calm
    -   Happy
    -   Sad
    -   Angry
    -   Fearful
    -   Disgust
    -   Surprised
-   Display the predicted emotion in a simple web interface.
-   Expose health-check and prediction endpoints through FastAPI.

## Tech Stack

-   **Language:** Python
-   **Audio processing:** Librosa, NumPy
-   **Deep learning:** TensorFlow, TensorFlow Hub, Keras
-   **Backend:** FastAPI
-   **Frontend:** HTML, CSS, JavaScript
-   **Model components:** YAMNet embeddings and a trained Keras
    classifier

## Project Structure

``` text
Audio_Emotion_Project/
├── backend/
│   └── app.py
├── frontend/                 # Present in the project; not used by the current root HTML flow
├── models/
│   ├── emotion_model.keras
│   └── yamnet/                # Local YAMNet SavedModel files
├── static/
│   ├── script.js
│   └── style.css
├── test_purposes/
├── index.html
├── requirements.txt
├── runtime.txt
├── speech_model.ipynb
├── .gitignore
└── README.md
```

The `models/yamnet/` directory is large and may not be included in a
source upload. On startup, the backend checks for
`models/yamnet/saved_model.pb`. If it is missing, the code downloads
YAMNet from TensorFlow Hub and saves it locally, so the first startup
requires network access and enough disk space.

## How It Works

1.  The user selects a WAV file in the browser.
2.  JavaScript sends the file as multipart form data to `POST /predict`.
3.  FastAPI saves the upload temporarily and validates that it is a
    non-empty WAV file.
4.  Librosa loads the audio at 16 kHz. The backend normalizes it and
    pads or trims it to `66,200` samples.
5.  YAMNet generates embeddings for the waveform. The backend averages
    the embeddings over time.
6.  The saved Keras model predicts the emotion class.
7.  The API returns the predicted label and confidence value; the
    frontend displays the emotion.

## Setup

### 1. Clone the repository

``` bash
git clone <YOUR_REPOSITORY_URL>
cd Audio_Emotion_Project
```

Replace `<YOUR_REPOSITORY_URL>` with your repository URL.

### 2. Create and activate a virtual environment

Python `3.10.16` is specified in `runtime.txt`.

**Windows (PowerShell):**

``` powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

``` bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install "uvicorn[standard]" python-multipart
```

The current `requirements.txt` lists the core audio, ML, and FastAPI
libraries. `uvicorn` is needed to run the ASGI app, and
`python-multipart` is needed for FastAPI file uploads; install them
explicitly unless you add them to `requirements.txt`.

TensorFlow and TensorFlow Hub can be platform-sensitive. If installation
fails, check the supported Python and operating-system versions for the
TensorFlow release you install.

### 4. Check the model files

Make sure this file exists:

``` text
models/emotion_model.keras
```

The YAMNet model is expected at:

``` text
models/yamnet/saved_model.pb
```

If the YAMNet SavedModel is absent, the backend attempts to download it
from TensorFlow Hub at startup. The first launch may therefore take
longer and requires internet access.

## Run the Application

From the **project root** (`Audio_Emotion_Project/`), run:

``` bash
uvicorn backend.app:app --reload
```

Then open:

-   Web interface: `http://127.0.0.1:8000/`
-   Health check: `http://127.0.0.1:8000/health`
-   Interactive API documentation: `http://127.0.0.1:8000/docs`

The application serves `index.html` from the project root and the
CSS/JavaScript files from `/static`.

## API

### `GET /health`

Checks whether the API is responding.

Example response:

``` json
{
  "status": "ok"
}
```

### `POST /predict`

Uploads a WAV audio file for emotion prediction. Send the request as
`multipart/form-data` using the field name `file`.

Example with `curl`:

``` bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@sample.wav"
```

Successful response format:

``` json
{
  "emotion": "happy",
  "confidence": 0.9234
}
```

The values above are illustrative; the actual emotion and confidence
depend on the input audio and model output. The confidence is returned
as a decimal rounded to four places.

The endpoint returns an HTTP `400` error for a non-WAV filename or an
empty upload. Other processing failures return an HTTP `500` response
with error details.

## Notes and Limitations

-   The frontend currently accepts `.wav` files only.
-   Predictions are limited to the eight emotion labels defined in
    `backend/app.py`.
-   The model output reflects the trained model and should not be
    treated as a definitive assessment of a speaker's emotional state.
-   The backend temporarily stores uploaded audio for processing and
    removes the temporary file afterward.
-   The current frontend displays the emotion label. Although the API
    also returns `confidence`, the supplied JavaScript does not
    currently display that value.
-   The project's `requirements.txt` includes Streamlit and Requests,
    but the current `index.html` + `static/script.js` interface
    communicates directly with FastAPI. Streamlit is not needed for this
    frontend flow.
-   For production deployment, configure resource limits, request-size
    limits, logging, dependency versions, and appropriate security
    controls.

## License

Add a license file if you intend to distribute this project publicly.

