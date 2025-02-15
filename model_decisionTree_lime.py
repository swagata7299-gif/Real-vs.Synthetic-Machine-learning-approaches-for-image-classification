import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import joblib
from pywt import dwt2
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

from lime import lime_image

explainer = lime_image.LimeImageExplainer()
print("LIME is working!")

# Load the trained model and scaler
catboost_model = joblib.load('decision_model.pkl')
scaler = joblib.load('scaler.pkl')

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
    feature_vector = scaler.transform([feature_vector])

    # Predict using the CatBoost model
    prediction = catboost_model.predict(feature_vector)[0]
    predicted_class = "Real" if prediction == 1 else "Fake"

    # Display the result
    result_label.config(text=f"Prediction: {predicted_class}")

    # Display the uploaded image on the GUI
    img = Image.open(file_path)
    img.thumbnail((250, 250))
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img  # Keep a reference to the image object

    # Generate LIME explanation
    generate_lime_explanation(image, feature_vector, file_path, predicted_class)

def generate_lime_explanation(image, feature_vector, file_path, predicted_class):
    def predict_proba(input_images):
        features_list = [extract_features(img) for img in input_images]
        valid_features = [f for f in features_list if f is not None]
        if not valid_features:
            return np.array([])
        valid_features = scaler.transform(valid_features)
        return catboost_model.predict_proba(valid_features)

    # Convert BGR image to RGB for LIME
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Custom segmentation function
    def custom_segmentation(image_array):
        from skimage.segmentation import slic
        return slic(image_array, n_segments=50, compactness=10, sigma=1)

    # Generate explanation with custom segmentation
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image_rgb,
        predict_proba,
        top_labels=2,
        hide_color=0,  # Use 0 (black) for hiding unimportant parts
        num_samples=50,  # Increase samples for better results
        segmentation_fn=custom_segmentation  # Apply custom segmentation
    )

    # Get explanation for the predicted label (show full segmentation)
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],  # Use the top predicted label
        positive_only=False,  # Show both positive and negative regions
        num_features=10,  # Display more key features
        hide_rest=False  # Do not hide unimportant areas
    )

    # Display the original image and LIME explanation side by side
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Original Image
    ax[0].imshow(image_rgb)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    # LIME Explanation with boundaries
    lime_result_image = mark_boundaries(temp / 255.0, mask)
    ax[1].imshow(lime_result_image)
    ax[1].set_title(f'LIME Explanation ({predicted_class})')
    ax[1].axis('off')

    # Show the result in a single plot
    plt.show()

# GUI setup
root = tk.Tk()
root.title("Image Prediction with LIME Explanation")
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
