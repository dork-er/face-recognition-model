import streamlit as st
import cv2 as cv
import numpy as np
import pickle
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from utils.preprocess import preprocess_faces
from io import BytesIO

# Titles and description
st.title("Face Recognition System")
st.write("Upload an image or a video to detect and recognize multiple faces.")

# Load Haarcascade and SVM model
haarcascade = cv.CascadeClassifier("models/haarcascade_frontalface_default.xml")
facenet = FaceNet()

# Load face embeddings and labels
data = np.load("models/faces_embeddings_144classes.npz")
embeddings = data['arr_0']
labels = data['arr_1']

# Fit the LabelEncoder with the labels
encoder = LabelEncoder()
encoder.fit(labels)

# Load the SVM model
with open("models/svm_model_144classes.pkl", "rb") as f:
    svm_model = pickle.load(f)

# Define confidence thresholds
CONFIDENCE_THRESHOLD = 0.85
UNKNOWN_THRESHOLD = 1.0
EXTREMELY_LOW_CONFIDENCE_THRESHOLD = 0.4

def recognize_faces(image):
    recognized_faces = []
    face_regions = preprocess_faces(image, haarcascade)

    for (x, y, w, h, face_img) in face_regions:
        # Resize and preprocess the face
        face_img = cv.resize(face_img, (160, 160))
        face_img = np.expand_dims(face_img, axis=0)
        embedding = facenet.embeddings(face_img)

        # Skip detection if embedding distance is very high (non-face)
        embedding_distances = np.linalg.norm(embedding - embeddings, axis=1)
        if np.min(embedding_distances) > UNKNOWN_THRESHOLD * 2:
            continue

        # Predict probabilities and class
        probabilities = svm_model.decision_function(embedding)
        max_prob = np.max(probabilities) if len(probabilities) > 0 else 0
        predicted_class = svm_model.predict(embedding)[0]
        predicted_name = encoder.inverse_transform([predicted_class])[0]

        # Skip extremely low-confidence predictions
        if max_prob < EXTREMELY_LOW_CONFIDENCE_THRESHOLD:
            continue

        # Determine if face is "Unknown"
        if max_prob < CONFIDENCE_THRESHOLD or np.min(embedding_distances) > UNKNOWN_THRESHOLD:
            predicted_name = "Unknown"

        # Append valid face data
        recognized_faces.append((x, y, w, h, predicted_name))

    # Annotate the image
    for (x, y, w, h, name) in recognized_faces:
        color = (0, 255, 0) if name != "Unknown" else (255, 0, 0)
        cv.rectangle(image, (x, y), (x + w, y + h), color, 7)
        cv.putText(image, name, (x, y-20), cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    return image, recognized_faces

def process_video(video_path):
    video = cv.VideoCapture(video_path)
    annotated_frames = []

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # Convert frame to RGB for processing
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        annotated_frame, _ = recognize_faces(frame_rgb)
        annotated_frames.append(cv.cvtColor(annotated_frame, cv.COLOR_RGB2BGR))

    video.release()
    return annotated_frames

# Upload image or video
uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"])


if uploaded_file is not None:
    if uploaded_file.type.startswith('image/'):
        # Process image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Processing...")

        annotated_image, recognized_faces = recognize_faces(image)
        st.image(annotated_image, caption="Processed Image", use_column_width=True)

        if recognized_faces:
            st.success("Recognized Faces:")
            for idx, face in enumerate(recognized_faces):
                st.write(f"{idx+1}: {face[-1]}")
        else:
            st.warning("No faces recognized.")

    elif uploaded_file.type.startswith('video/'):
        # Process video
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.read())

        st.video(temp_video_path)

        st.write("Processing video...")
        annotated_frames = process_video(temp_video_path)

        # Save annotated frames to a video
        output_video_path = "annotated_video.mp4"
        height, width, _ = annotated_frames[0].shape
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(output_video_path, fourcc, 20.0, (width, height))

        for frame in annotated_frames:
            out.write(frame)

        out.release()

        # Display the annotated video
        st.video(output_video_path)
