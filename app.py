import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Load your trained model
model = load_model('cnn_model.h5')

# Define the labels
labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Function to preprocess the image
def preprocess_image(image, target_size):
    if image.mode != 'L':
        image = image.convert('L')
    image = image.resize(target_size)
    image = np.array(image)
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit app
st.title("Brain MRI Classification")

st.write("Upload an MRI image to classify it into one of the following categories: glioma, meningioma, notumor, pituitary")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    processed_image = preprocess_image(image, target_size=(200, 200))
    
    # Make prediction
    prediction = model.predict(processed_image)
    predicted_class = labels[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    
    # Display the result
    st.write(f"Predicted Class: **{predicted_class}**")
    st.write(f"Confidence: **{confidence:.2f}**")