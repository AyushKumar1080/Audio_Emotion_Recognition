
"use strict";

const audioForm = document.getElementById("audioForm");
const audioFile = document.getElementById("audioFile");
const status = document.getElementById("status");
const result = document.getElementById("result");
const emotion = document.getElementById("emotion");

// Predict emotion when the form is submitted
audioForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const file = audioFile.files[0];

    if (!file) {
        status.textContent = "Please select an audio file.";
        return;
    }

    if (!file.name.toLowerCase().endsWith(".wav")) {
        status.textContent = "Please upload a WAV file.";
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    const button = audioForm.querySelector('button[type="submit"]');

    button.disabled = true;
    status.textContent = "Analyzing audio...";
    result.hidden = true;

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Prediction failed.");
        }

        emotion.textContent =
            data.emotion || data.prediction || "Unknown";

        result.hidden = false;
        status.textContent = "Prediction completed successfully.";

    } catch (error) {
        status.textContent = `Error: ${error.message}`;
        console.error("Prediction error:", error);

    } finally {
        button.disabled = false;
    }
});
