import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import os
from transformers import pipeline

# Initialize Hugging Face model for text generation
text_generator = pipeline("text-generation", model="gpt2")

# Load pre-trained models
cnn_model = load_model('CNN.model')  # Replace with the correct path to your CNN model
filename = 'model.sav'
ml_model = pickle.load(open(filename, 'rb'))

# Dataset directory and class names
data_dir = "dataset"
class_names = os.listdir(data_dir)

def generate_ai_advice(pest_result, crop_prediction):
    try:
        # Refined prompt for better results
        prompt = f"""
        Detect pest '{pest_result}' on the leaf and recommend crop '{crop_prediction}'.
        1. Explain the pest and its effects on crops.
        2. Suggest control methods for the pest.
        3. Provide tips for growing '{crop_prediction}'.
        """
        # Generate text with increased length and avoid repetition
        response = text_generator(
            prompt, 
            max_length=300,  # Increased length for detailed responses
            num_return_sequences=1, 
            no_repeat_ngram_size=3  # Avoid repetitive phrases
        )
        advice = response[0]["generated_text"]
        # Clean up unwanted characters
        clean_advice = ''.join(char for char in advice if char.isprintable())
        return clean_advice
    except Exception as e:
        print(e)
        return f"Error generating AI advice: {e}"

def classify_image_and_get_crop():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Load and preprocess the image
        img = cv2.imread(file_path)
        img = cv2.medianBlur(img, 1)
        img = cv2.resize(img, (50, 50))  # Resize to CNN input dimensions
        img_expanded = np.expand_dims(img, axis=0)
        img_normalized = img_expanded / 255.0

        # Predict the pest
        predictions = cnn_model.predict(img_normalized)
        class_index = np.argmax(predictions)
        class_label = class_names[class_index]
        
        pest_result = class_label

        # Collect inputs for crop prediction
        try:
            a = int(n_input.get())
            b = int(p_input.get())
            c = int(k_input.get())
            d = float(temp_input.get())
            e = float(humidity_input.get())
            f = float(ph_input.get())
            g = float(rainfall_input.get())
            crop_inputs = [[a, b, c, d, e, f, g]]
            crop_prediction = ml_model.predict(crop_inputs)[0]
            
            # Generate advice
            advice = generate_ai_advice(pest_result, crop_prediction)
            
            # Display results and advice
            result_label.config(
                text=f"Result: {pest_result}\nCrop Recommended: {crop_prediction}\n\nAI Advice:\n{advice}",
                justify="left",  # Align advice text for readability
                wraplength=600   # Wrap text to fit the UI window
            )
        except ValueError:
            result_label.config(text="Error: Invalid inputs for crop prediction")


# Tkinter setup
root = tk.Tk()
root.title("Pest and Crop Predictor")

# Title Label
title_label = tk.Label(root, text="Pest Detection and Crop Recommendation", font=("Helvetica", 18))
title_label.pack(pady=10)

# Input Fields for Crop Prediction
input_frame = tk.Frame(root)
input_frame.pack(pady=10)

tk.Label(input_frame, text="N").grid(row=0, column=0, padx=5, pady=5)
n_input = tk.Entry(input_frame)
n_input.grid(row=0, column=1, padx=5, pady=5)

tk.Label(input_frame, text="P").grid(row=1, column=0, padx=5, pady=5)
p_input = tk.Entry(input_frame)
p_input.grid(row=1, column=1, padx=5, pady=5)

tk.Label(input_frame, text="K").grid(row=2, column=0, padx=5, pady=5)
k_input = tk.Entry(input_frame)
k_input.grid(row=2, column=1, padx=5, pady=5)

tk.Label(input_frame, text="Temperature").grid(row=3, column=0, padx=5, pady=5)
temp_input = tk.Entry(input_frame)
temp_input.grid(row=3, column=1, padx=5, pady=5)

tk.Label(input_frame, text="Humidity").grid(row=4, column=0, padx=5, pady=5)
humidity_input = tk.Entry(input_frame)
humidity_input.grid(row=4, column=1, padx=5, pady=5)

tk.Label(input_frame, text="pH").grid(row=5, column=0, padx=5, pady=5)
ph_input = tk.Entry(input_frame)
ph_input.grid(row=5, column=1, padx=5, pady=5)

tk.Label(input_frame, text="Rainfall").grid(row=6, column=0, padx=5, pady=5)
rainfall_input = tk.Entry(input_frame)
rainfall_input.grid(row=6, column=1, padx=5, pady=5)

# Classify Button
classify_button = tk.Button(root, text="Select Image & Predict", command=classify_image_and_get_crop)
classify_button.pack(pady=10)

# Result Label
result_label = tk.Label(root, text="", font=("Helvetica", 14))
result_label.pack(pady=20)

# Quit Button
quit_button = tk.Button(root, text="Quit", command=root.destroy)
quit_button.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
