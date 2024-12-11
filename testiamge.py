import tkinter as tk
from tkinter import filedialog
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import os

# Load your pre-trained model ('CNN.model')
model = load_model('CNN.model')  # Replace with the correct path to your CNN.model
data_dir = "dataset"
class_names = os.listdir(data_dir)

# Function to classify an image
def classify_image():
    # Open a file dialog to select an image
    file_path = filedialog.askopenfilename()
    if file_path:
        # Load and preprocess the selected image
        img = cv2.imread(file_path)
        img = cv2.medianBlur(img, 1)
        img = cv2.resize(img, (50, 50))  # Adjust the size if needed
        img_expanded = np.expand_dims(img, axis=0)
        img_normalized = img_expanded / 255.0  # Normalize the image

        # Make predictions
        predictions = model.predict(img_normalized)
        class_index = np.argmax(predictions)
        class_label = class_names[class_index]
        
        # Check if pest is detected
        if class_label != "Normal_Leaf":
            # Convert the image to HSV color space for green detection
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Define the range for green color in HSV
            lower_green = np.array([35, 40, 40])
            upper_green = np.array([85, 255, 255])
            
            # Create a mask to remove green areas
            mask = cv2.inRange(hsv_img, lower_green, upper_green)
            masked_img = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
            
            # Display the masked image showing only non-green areas
            cv2.imshow("Pest Detected - Non-Green Areas Only", masked_img)
            cv2.waitKey(0)  # Wait until a key is pressed
            cv2.destroyAllWindows()
            result_text = "Pest Detected"
        else:
            result_text = "No pest"
        
        # Display the result in the tkinter window
        result_label.config(text=f'Result: {result_text}')

# Create a tkinter window
root = tk.Tk()
root.title("Classifier")

# Create a label for the title
title_label = tk.Label(root, text="Classification", font=("Helvetica", 20))
title_label.pack(pady=20)

# Create a label to display the result
result_label = tk.Label(root, text="", font=("Helvetica", 16))
result_label.pack(pady=20)

# Create a button to select an image
classify_button = tk.Button(root, text="Select Image", command=classify_image)
classify_button.pack()

# Create a quit button
quit_button = tk.Button(root, text="Quit", command=root.destroy)
quit_button.pack()

# Start the tkinter main loop
root.mainloop()
