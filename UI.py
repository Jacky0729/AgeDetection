import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import requests
import os
import cv2
from PIL import Image

# Function to download the model from GitHub
@st.cache_resource(show_spinner=True)
def download_model_from_github(model_name):
    url = f'https://github.com/Jacky0729/AgeDetection/raw/main/{model_name}.keras'  # Adjust this URL if needed
    response = requests.get(url)

    if response.status_code != 200:
        st.error(f"Failed to download {model_name}.keras. Status code: {response.status_code}")
        return None

    model_path = f'{model_name}.keras'
    with open(model_path, 'wb') as file:
        file.write(response.content)

    if not os.path.exists(model_path):
        st.error(f"File not found after download: {model_path}")
        return None

    return model_path

# Load the model after downloading
def load_trained_model_from_github(model_name):
    model_path = download_model_from_github(model_name)
    if model_path is not None:
        try:
            model = load_model(model_path)
            return model
        except Exception as e:
            st.error(f"Error loading model from {model_path}: {e}")
            return None
    else:
        return None

# Load model from GitHub
model = load_trained_model_from_github('multi_output_model')

# Preprocess the image
def preprocess_image(image):
    target_size = (200, 200)  # Match the model input size
    img = image.resize(target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize to [0, 1]
    return img_array

# Detect faces using OpenCV
def detect_face(image):
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# Streamlit app title
st.title("Age, Gender, and Race Classification")

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    faces = detect_face(image)

    if len(faces) == 0:
        st.write("No face detected in the image.")
    else:
        for (x, y, w, h) in faces:
            face_img = image.crop((x, y, x+w, y+h))
            st.image(face_img, caption="Detected Face", use_column_width=True)

            img_array = preprocess_image(face_img)

            if model:
                # Predict age, gender, and race
                predictions = model.predict(img_array)

                age_prediction, gender_prediction, race_prediction = predictions

                # Define the class mappings for the outputs
                age_groups = ['0-8', '9-18', '19-39', '40-59', '60+']
                gender_classes = ['Male', 'Female']
                race_classes = ['White', 'Black', 'Asian', 'Indian']

                predicted_age = age_groups[np.argmax(age_prediction)]
                predicted_gender = gender_classes[np.argmax(gender_prediction)]
                predicted_race = race_classes[np.argmax(race_prediction)]

                st.write(f"Predicted Age Group: {predicted_age}")
                st.write(f"Predicted Gender: {predicted_gender}")
                st.write(f"Predicted Race: {predicted_race}")
            else:
                st.error("Model could not be loaded. Please check the file.")
