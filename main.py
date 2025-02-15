import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from pywt import dwt2
from sklearn.preprocessing import StandardScaler
from lime import lime_image
import matplotlib.pyplot as plt

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

# Load dataset
def load_data(base_path):
    features, labels = [], []
    for label in ['real', 'fake']:
        folder_path = os.path.join(base_path, label)
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path)
            if image is not None:
                feature_vector = extract_features(image)
                if feature_vector is not None:  # Only append valid feature vectors
                    features.append(feature_vector)
                    labels.append(1 if label == 'real' else 0)  # 1 for real, 0 for fake
    return np.array(features), np.array(labels)

# Load training data
train_base_path = r"C:\Desktop\ML_Implementation\data(Final_ML)\train"
print("Loading training data...")
X_train, y_train = load_data(train_base_path)
print(f"Loaded {len(X_train)} training samples.")

# Load test data
test_base_path = r"C:\Desktop\ML_Implementation\data(Final_ML)\test"
print("Loading test data...")
X_test, y_test = load_data(test_base_path)
print(f"Loaded {len(X_test)} test samples.")

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Initialize the CatBoost Classifier
catboost_model = CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.1, random_seed=42, verbose=0)

# Fit the model to the training data
catboost_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = catboost_model.predict(X_test)

# Generate performance metrics
classification_rep = classification_report(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

# Save results as attributes in the model
catboost_model.classification_report_ = classification_rep
catboost_model.confusion_matrix_ = confusion_mat

# Print performance metrics
print("Classification Report:\n", classification_rep)
print("Confusion Matrix:\n", confusion_mat)

# Save the trained CatBoost model and scaler
joblib.dump(catboost_model, 'catboost_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
