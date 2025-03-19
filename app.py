import streamlit as st
import pandas as pd
import numpy as np

from sklearn.datasets import fetch_openml
import cv2 
from PIL import Image
from io import BytesIO

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from streamlit_drawable_canvas import st_canvas

st.header("App created by Priyadarsini(Priya)")

# Load MNIST dataset and train a simple logistic regression model
mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
X = mnist["data"]
y = mnist["target"].astype(np.uint8)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train  = X_train / 255.0
y_test = y_test / 255.0


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA(n_components=0.99)  # Retain 99% of variance
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)


#Train Logistic Regression model
log_reg = LogisticRegression(max_iter=2000)
log_reg.fit(X_train_pca, y_train)

# Creating the streamlit application

nav = st.sidebar.radio("Navigation Menu",["Purpose", "Prediction"])

if nav == "Purpose":
    st.title("Streamlit - My First Web Application")
    st.header("Purpose")
    st.write(""" The purpose is the model will predict a number from a hand drawn digit image.""" )


elif nav == "Prediction":
    st.header("Prediction")
    st.write("Draw a digit in the box below, and the model will predict which digit it is.")

    # Create a drawable canvas
    canvas_result = st_canvas(
        fill_color="black",  # Background color
        stroke_width=20,  # Thickness of the stroke
        stroke_color="white",  # Stroke color (white for digit)
        background_color="black",  # Black background
        width=280,  # Larger canvas to draw on
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        # Convert canvas image to grayscale
        img = cv2.cvtColor(canvas_result.image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
        
        # Resize and preprocess to 28x28
        img = cv2.resize(img, (28, 28))
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Display the processed image
        st.image(img, caption="Processed Image", width=150)

        # Normalize and reshape for model prediction
        img = img / 255.0
        img = img.reshape(1, -1)

        # Standardize and apply PCA
        img = scaler.transform(img)
        img_pca = pca.transform(img)

        # Predict digit
        prediction = log_reg.predict(img_pca)[0]
        st.write(f"Predicted Digit: {prediction}")
