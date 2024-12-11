from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from transformers import pipeline
import os

app = Flask(__name__)
app.secret_key = "secret_key"

# Initialize Hugging Face model for text generation
text_generator = pipeline("text-generation", model="gpt2")

# Load pre-trained models
cnn_model = load_model('CNN.model')  # Replace with your actual CNN model path
ml_model = pickle.load(open('model.sav', 'rb'))

# Dataset directory and class names
data_dir = "dataset"
class_names = os.listdir(data_dir)

def generate_ai_advice(pest_result, crop_prediction):
    try:
        prompt = f"""
        Detect pest '{pest_result}' on the leaf and recommend crop '{crop_prediction}'.
        1. Explain the pest and its effects on crops.
        2. Suggest control methods for the pest.
        3. Provide tips for growing '{crop_prediction}'.
        """
        response = text_generator(
            prompt, max_length=300, num_return_sequences=1, no_repeat_ngram_size=3
        )
        advice = response[0]["generated_text"]
        return ''.join(char for char in advice if char.isprintable())
    except Exception as e:
        return f"Error generating AI advice: {e}"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get input data
        try:
            n = int(request.form['N'])
            p = int(request.form['P'])
            k = int(request.form['K'])
            temp = float(request.form['Temperature'])
            humidity = float(request.form['Humidity'])
            ph = float(request.form['pH'])
            rainfall = float(request.form['Rainfall'])
        except ValueError:
            return render_template("index.html", error="Invalid input data")

        # File upload
        if "image" not in request.files or request.files["image"].filename == "":
            return render_template("index.html", error="Please upload an image")

        file = request.files["image"]
        filepath = os.path.join("static", file.filename)
        file.save(filepath)

        # Process image
        img = cv2.imread(filepath)
        img = cv2.medianBlur(img, 1)
        img = cv2.resize(img, (50, 50))
        img_expanded = np.expand_dims(img, axis=0)
        img_normalized = img_expanded / 255.0

        # Predict pest
        predictions = cnn_model.predict(img_normalized)
        pest_result = class_names[np.argmax(predictions)]

        # Predict crop
        crop_inputs = [[n, p, k, temp, humidity, ph, rainfall]]
        crop_prediction = ml_model.predict(crop_inputs)[0]

        # Generate advice
        advice = generate_ai_advice(pest_result, crop_prediction)

        return render_template(
            "index.html",
            pest_result=pest_result,
            crop_prediction=crop_prediction,
            advice=advice,
            image_url=filepath,
        )
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
