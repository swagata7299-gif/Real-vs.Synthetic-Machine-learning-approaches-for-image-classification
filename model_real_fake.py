import os
import numpy as np
import cv2
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import joblib
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from pywt import dwt2

# Function for extracting features
def extract_features(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # DCT
    dct_features = cv2.dct(np.float32(gray) / 255.0)
    dct_features = dct_features.flatten()[:100]  # Top 100 coefficients

    # Pad or trim to make it exactly 100 coefficients if necessary
    if len(dct_features) < 100:
        dct_features = np.pad(dct_features, (0, 100 - len(dct_features)), 'constant')
    else:
        dct_features = dct_features[:100]

    # Wavelet Transform
    coeffs2 = dwt2(gray, 'haar')
    cA, (cH, cV, cD) = coeffs2
    wavelet_features = np.concatenate([cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()])[:100]

    # Pad or trim to make it exactly 100 coefficients if necessary
    if len(wavelet_features) < 100:
        wavelet_features = np.pad(wavelet_features, (0, 100 - len(wavelet_features)), 'constant')
    else:
        wavelet_features = wavelet_features[:100]

    # FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    epsilon = 1e-10  # To avoid log(0)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + epsilon)
    fft_features = magnitude_spectrum.flatten()[:100]  # Top 100 coefficients

    # Pad or trim to make it exactly 100 coefficients if necessary
    if len(fft_features) < 100:
        fft_features = np.pad(fft_features, (0, 100 - len(fft_features)), 'constant')
    else:
        fft_features = fft_features[:100]

    # Combine features
    features = np.concatenate([dct_features, wavelet_features, fft_features])

    # Check for NaN or infinite values
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        return None  # Return None for invalid features

    return features

# Load the saved model and scaler
decision_model = joblib.load('decision_model.pkl')
scaler1 = joblib.load('scaler1.pkl')

# Function to predict the image label
def predict_image(image_path):
    # Read and preprocess the image
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # Extract features from the image
    features = extract_features(image)
    if features is None:
        return None

    # Scale the features using the fitted scaler
    features_scaled = scaler1.transform([features])

    # Use the CatBoost model to predict the label
    prediction = decision_model.predict(features_scaled)
    
    # Debugging: Print the prediction output
    print(f"Prediction Output: {prediction}")

    if prediction == 1:
        return 'real'
    elif prediction == 0:
        return 'fake'
    else:
        return 'Error: Unknown prediction'

# Function to open file dialog and predict the image
def upload_image():
    # Open file dialog to select image
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=(("JPEG files", "*.jpg;*.jpeg"), ("All files", "*.*")))
    if file_path:
        # Display the selected image
        image = Image.open(file_path)
        image = image.resize((250, 250), Image.Resampling.LANCZOS)  # Updated line for resizing
        photo = ImageTk.PhotoImage(image)

        # Display image in the window
        label_image.config(image=photo)
        label_image.image = photo

        # Predict and display the result
        prediction = predict_image(file_path)
        if prediction:
            label_prediction.config(text=f"Prediction: {prediction}")
        else:
            label_prediction.config(text="Error: Invalid image")

# Create the main window
root = Tk()
root.title("Image Classification - Real or Fake")

# Create a button to upload images
button_upload = Button(root, text="Upload Image", command=upload_image)
button_upload.pack(pady=20)

# Create a label to display the uploaded image
label_image = Label(root)
label_image.pack(pady=20)

# Create a label to display the prediction result
label_prediction = Label(root, text="Prediction: ", font=("Helvetica", 16))
label_prediction.pack(pady=20)

# Start the Tkinter event loop
root.mainloop()


# Function for extracting features
def extract_features(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # DCT
    dct_features = cv2.dct(np.float32(gray) / 255.0)
    dct_features = dct_features.flatten()[:100]  # Top 100 coefficients

    # Pad or trim to make it exactly 100 coefficients if necessary
    if len(dct_features) < 100:
        dct_features = np.pad(dct_features, (0, 100 - len(dct_features)), 'constant')
    else:
        dct_features = dct_features[:100]

    # Wavelet Transform
    coeffs2 = dwt2(gray, 'haar')
    cA, (cH, cV, cD) = coeffs2
    wavelet_features = np.concatenate([cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()])[:100]

    # Pad or trim to make it exactly 100 coefficients if necessary
    if len(wavelet_features) < 100:
        wavelet_features = np.pad(wavelet_features, (0, 100 - len(wavelet_features)), 'constant')
    else:
        wavelet_features = wavelet_features[:100]

    # FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    epsilon = 1e-10  # To avoid log(0)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + epsilon)
    fft_features = magnitude_spectrum.flatten()[:100]  # Top 100 coefficients

    # Pad or trim to make it exactly 100 coefficients if necessary
    if len(fft_features) < 100:
        fft_features = np.pad(fft_features, (0, 100 - len(fft_features)), 'constant')
    else:
        fft_features = fft_features[:100]

    # Combine features
    features = np.concatenate([dct_features, wavelet_features, fft_features])

    # Check for NaN or infinite values
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        return None  # Return None for invalid features

    return features

# Load the saved model and scaler
catboost_model = joblib.load('catboost_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to predict the image label
def predict_image(image_path):
    # Read and preprocess the image
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # Extract features from the image
    features = extract_features(image)
    if features is None:
        return None

    # Scale the features using the fitted scaler
    features_scaled = scaler.transform([features])

    # Use the CatBoost model to predict the label
    prediction = catboost_model.predict(features_scaled)
    
    # Debugging: Print the prediction output
    print(f"Prediction Output: {prediction}")

    if prediction == 1:
        return 'real'
    elif prediction == 0:
        return 'fake'
    else:
        return 'Error: Unknown prediction'

# Function to open file dialog and predict the image
def upload_image():
    # Open file dialog to select image
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=(("JPEG files", "*.jpg;*.jpeg"), ("All files", "*.*")))
    if file_path:
        # Display the selected image
        image = Image.open(file_path)
        image = image.resize((250, 250), Image.Resampling.LANCZOS)  # Updated line for resizing
        photo = ImageTk.PhotoImage(image)

        # Display image in the window
        label_image.config(image=photo)
        label_image.image = photo

        # Predict and display the result
        prediction = predict_image(file_path)
        if prediction:
            label_prediction.config(text=f"Prediction: {prediction}")
        else:
            label_prediction.config(text="Error: Invalid image")

# Create the main window
root = Tk()
root.title("Image Classification - Real or Fake")

# Create a button to upload images
button_upload = Button(root, text="Upload Image", command=upload_image)
button_upload.pack(pady=20)

# Create a label to display the uploaded image
label_image = Label(root)
label_image.pack(pady=20)

# Create a label to display the prediction result
label_prediction = Label(root, text="Prediction: ", font=("Helvetica", 16))
label_prediction.pack(pady=20)

# Start the Tkinter event loop
root.mainloop()