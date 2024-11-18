import streamlit as st
import cv2 as cv
import numpy as np
import pickle
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from utils.preprocess import preprocess_faces

# Initialize models
st.title("Face Recognition System")
st.write("Upload an image to detect and recognize multiple faces.")

# Load Haarcascade and SVM model
haarcascade = cv.CascadeClassifier("models/haarcascade_frontalface_default.xml")
facenet = FaceNet()

# Load SVM model and encoder
try:
    svm_model = pickle.load(open("models/svm_model.pkl", 'rb'))
    encoder = LabelEncoder()
    encoder.classes_ = svm_model.classes_
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

def recognize_faces(image):
    # Detect and preprocess all faces in the image
    face_regions = preprocess_faces(image, haarcascade)
    if len(face_regions) == 0:
        return image, []

    recognized_faces = []
    for (x, y, w, h, face_img) in face_regions:
        # Resize and preprocess
        face_img = cv.resize(face_img, (160, 160))
        face_img = np.expand_dims(face_img, axis=0)
        embedding = facenet.embeddings(face_img)

        # Predict
        prediction = svm_model.predict(embedding)
        predicted_name = encoder.inverse_transform(prediction)[0]
        if predicted_name:
            predicted_name = str(predicted_name).replace('_', ' ')
        else:
            predicted_name = "Unknown"
        recognized_faces.append((x, y, w, h, predicted_name))

    # Annotate the image
    for (x, y, w, h, name) in recognized_faces:
        cv.rectangle(image, (x, y), (x+w, y+h), (255, 0, 255), 3)
        cv.putText(image, name, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return image, recognized_faces

# Upload and process image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv.imdecode(file_bytes, cv.IMREAD_COLOR)

    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    annotated_image, names = recognize_faces(image)
    st.image(annotated_image, caption="Processed Image", use_column_width=True)

    if names:
        st.success("Recognized Faces:")
        for idx, name in enumerate(names):
            st.write(f"{idx+1}: {name[-1]}")  # Name is the last item in the tuple
    else:
        st.warning("No faces recognized.")
