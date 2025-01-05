import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st


working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/brain_tumor_detection_model.h5"
# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# loading the class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))


# Function to Load and Preprocess the Image using Pillow

import cv2
import numpy as np

def load_and_preprocess_image(uploaded_image, target_size=(150, 150)):
    img = Image.open(uploaded_image)  # Read the image as a PIL object
    img = img.resize(target_size)  # Resize to 150x150
    img_array = np.array(img)
    img_array = img_array.reshape(1, target_size[0], target_size[1], 3)  # Reshape to (1, 150, 150, 3)
    return img_array



# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    indices = predictions.argmax()
    indices=str(indices)
    predicted_class_name = class_indices[indices]
    return predicted_class_name


# Streamlit App
st.title('Brain Tumor Detection')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image and predict the class
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {str(prediction)}')
