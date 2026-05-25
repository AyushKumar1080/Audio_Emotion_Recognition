# Audio Emotion Recognition using YAMNet

This project predicts human emotions from audio (`.wav`) files using a Deep Learning approach. The model is trained on the RAVDESS dataset and is capable of classifying speech into **8 different emotions**.

## Dataset

The dataset used in this project is the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset.

Dataset Link:
[RAVDESS Dataset](https://zenodo.org/records/1188976?utm_source=chatgpt.com)

The dataset contains the following emotions:

1. Neutral
2. Calm
3. Happy
4. Sad
5. Angry
6. Fearful
7. Disgust
8. Surprised

## Model Used

This project uses the pretrained YAMNet Deep Learning model for audio feature extraction. YAMNet is trained for audio understanding tasks and generates rich audio embeddings that help improve emotion classification performance.

## Audio Processing & Feature Extraction

Audio loading and preprocessing are performed using the [Librosa](https://librosa.org?utm_source=chatgpt.com) library. The preprocessing pipeline includes:

* Audio loading from `.wav` files
* Signal normalization
* Padding and trimming of audio samples
* Feature extraction using YAMNet embeddings

This project indirectly uses the concept of **Mel Spectrograms**, because YAMNet internally converts audio waveforms into **Log-Mel Spectrogram representations** before generating embeddings.

## Model Improvement Techniques

To improve model performance and reduce overfitting, several deep learning techniques were implemented:

* Dropout
* Batch Normalization
* Regularization

## Data Augmentation

Data augmentation techniques were used to create additional training samples from existing audio data, helping increase dataset size and improve model generalization.

The augmentation techniques include:

* Noise addition
* Time shifting

These techniques help the model become more robust and improve prediction accuracy.

---

This project demonstrates the application of Deep Learning and audio signal processing for Speech Emotion Recognition (SER).
