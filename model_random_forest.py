import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import joblib
from pywt import dwt2
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
random_model = joblib.load('randomForest_model.pkl')
scaler2 = joblib.load('scaler2.pkl')

# Function for extracting features from the uploaded image
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # DCT
    dct_features = cv2.dct(np.float32(gray) / 255.0)
    dct_features = dct_features.flatten()[:100]

    if len(dct_features) < 100:
        dct_features = np.pad(dct_features, (0, 100 - len(dct_features)), 'constant')
    else:
        dct_features = dct_features[:100]

    # Wavelet Transform
    coeffs2 = dwt2(gray, 'haar')
    cA, (cH, cV, cD) = coeffs2
    wavelet_features = np.concatenate([cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()])[:100]

    if len(wavelet_features) < 100:
        wavelet_features = np.pad(wavelet_features, (0, 100 - len(wavelet_features)), 'constant')
    else:
        wavelet_features = wavelet_features[:100]

    # FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    epsilon = 1e-10
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + epsilon)
    fft_features = magnitude_spectrum.flatten()[:100]

    if len(fft_features) < 100:
        fft_features = np.pad(fft_features, (0, 100 - len(fft_features)), 'constant')
    else:
        fft_features = fft_features[:100]

    features = np.concatenate([dct_features, wavelet_features, fft_features])

    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        return None

    return features

# Function for processing uploaded image and making prediction
def predict_image():
    # Open file dialog to select image
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])

    if not file_path:
        return

    # Load and process the image
    image = cv2.imread(file_path)
    if image is None:
        messagebox.showerror("Error", "Unable to load image.")
        return

    feature_vector = extract_features(image)
    if feature_vector is None:
        messagebox.showerror("Error", "Error extracting features from the image.")
        return

    # Scale the features
    feature_vector = scaler2.transform([feature_vector])

    # Predict using the CatBoost model
    prediction = random_model.predict(feature_vector)[0]
    predicted_class = "Real" if prediction == 1 else "Fake"

    # Display the result
    result_label.config(text=f"Prediction: {predicted_class}")

    # Display the uploaded image on the GUI
    img = Image.open(file_path)
    img.thumbnail((250, 250))
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img  # Keep a reference to the image object

# GUI setup
root = tk.Tk()
root.title("Image Prediction")
root.geometry("600x400")

# Upload button
upload_button = tk.Button(root, text="Upload Image", command=predict_image)
upload_button.pack(pady=20)

# Label to display the result
result_label = tk.Label(root, text="Prediction: ", font=("Arial", 16))
result_label.pack(pady=10)

# Label to display the uploaded image
image_label = tk.Label(root)
image_label.pack(pady=20)

# Run the GUI
root.mainloop()
