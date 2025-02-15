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
import joblib

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


import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Initialize the Random Forest Classifier with chosen parameters
random_forest = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)

# Fit the model to the training data
random_forest.fit(X_train, y_train)

# Make predictions on the test set
y_pred = random_forest.predict(X_test)

# Performance Metrics
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Save the trained CatBoost model and scaler
joblib.dump(random_forest, 'randomForest_model.pkl')
joblib.dump(scaler, 'scaler2.pkl')