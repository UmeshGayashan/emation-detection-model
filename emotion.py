import streamlit as st
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import cvlib as cv

# Function to perform emotion detection on a single image
def detect_emotion(image):
    # Apply face detection
    faces, confidences = cv.detect_face(image)

    for face, confidence in zip(faces, confidences):
        # Get corner points of face rectangle
        startX, startY, endX, endY = face[0], face[1], face[2], face[3]

        # Draw rectangle over face
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Crop the detected face region
        face_crop = np.copy(image[startY:endY, startX:endX])

        if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            continue

        # Preprocessing for emotion detection model
        face_crop = cv2.resize(face_crop, (48, 48))
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # Apply emotion detection model
        conf = emotion_model.predict(face_crop)[0]

        # Get label with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]

        accuracy = f"{conf[idx]*100:.2f}%"

        return label, accuracy, image

# Streamlit app
st.header('Emotion Detection CNN Model')

# Load the pre-trained emotion detection model
emotion_model = load_model('emotion_detection_model_100epochs.h5')
classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# File upload for images
uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Perform emotion detection on the image
    label, accuracy, annotated_image = detect_emotion(image)
    
    # Display the image with the desired size
    st.image(annotated_image, channels="BGR", width=300)
  
    st.write(f"<span style='font-size: x-large; color: yellow;'>Emotion: {label} - {accuracy}.</span>", unsafe_allow_html=True)
