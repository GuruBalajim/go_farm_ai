import tkinter as tk
from tkinter import messagebox
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import os
import urllib.request

# Load your pre-trained model ('CNN.model')
model = load_model('CNN.model')  # Replace with the correct path to your CNN.model
data_dir = "dataset"
class_names = os.listdir(data_dir)

# Function to classify an image
def classify_image(img):
    img = cv2.medianBlur(img, 1)
    img = cv2.resize(img, (50, 50))  # Adjust the size if needed
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the image
    img = np.asarray(img)  # Convert to numpy array

    # Make predictions
    predictions = model.predict(img)
    class_index = np.argmax(predictions)
    class_label = class_names[class_index]
    if class_label != "Normal_Leaf":
         result_text = "Pest Detected"
    else:
        result_text = "No pest"
        
         

    # Display the result
    
    result_label.config(text=f'Result: {result_text}')
    messagebox.showinfo("Result", f"Result: {result_text}")

    # Check if class_index is 1 or 3
    if class_index in [1, 3]:
        # Trigger an HTTP request to a specific URL
        try:
            url = "https://api.thingspeak.com/update?api_key=C6TYSSCZS42F7QCI&field3=COIMBATORE&field4=ROAD_DAMAGED"
            urllib.request.urlopen(url)
            messagebox.showinfo("Alert", " HTTP request sent.")
        except Exception as e:
            messagebox.showerror("Error", f"Error sending HTTP request: {str(e)}")

# Function to start video classification
def start_video_classification():
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam

    while True:
        ret, frame = cap.read()
        cv2.imshow('Webcam', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            classify_image(frame)

        if key == ord('c'):
            result_label.config(text="")
            messagebox.showinfo("Result", "Classification reset.")

        if key == ord('e'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Create a tkinter window
root = tk.Tk()
root.title("Classifier")

# Create a label for the title
title_label = tk.Label(root, text="Classification", font=("Helvetica", 20))
title_label.pack(pady=20)

# Create a label to display the result
result_label = tk.Label(root, text="", font=("Helvetica", 16))
result_label.pack(pady=20)

# Create a button to start video classification
start_button = tk.Button(root, text="Start Video Classification", command=start_video_classification)
start_button.pack()

# Create a quit button
quit_button = tk.Button(root, text="Quit", command=root.destroy)
quit_button.pack()

# Start the tkinter main loop
root.mainloop()
