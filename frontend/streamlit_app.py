import streamlit as st
import requests

st.title("Speech Emotion Recognition")

st.write("Upload a WAV file")

uploaded=st.file_uploader("Upload audio",type=['wav'])

if uploaded:
    st.audio(uploaded)
    
    if st.button("Predict Emotion"):
        files={
            'file':(uploaded.name,uploaded,'audio/wav')
        }
        response=requests.post(
            'http://localhost:8000/predict',
            files=files
        )
        result=response.json()

        st.success(f"Emotion: {result['emotion']}")

        st.write(f"Confidence: {result['confidence']:.2%}")
