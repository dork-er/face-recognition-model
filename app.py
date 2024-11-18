import streamlit as st
import cv2 as cv
import numpy as np
import pickle
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from utils.preprocess import preprocess_faces
from fastapi import FastAPI, File, UploadFile
from starlette.responses import JSONResponse
import uvicorn
from io import BytesIO

# Initialize FastAPI app
api_app = FastAPI()

# Initialize models
st.title("Face Recognition System")
st.write("Upload an image to detect and recognize multiple faces.")

# Load Haarcascade and SVM model
haarcascade = cv.CascadeClassifier("models/haarcascade_frontalface_default.xml")
facenet = FaceNet()

# Load face embeddings and labels
data = np.load("models/faces_embeddings.npz")
embeddings = data['arr_0']
labels = data['arr_1']

# Fit the LabelEncoder with the labels
encoder = LabelEncoder()
encoder.fit(labels)

# Load the SVM model
with open("models/svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)

# Define confidence thresholds
CONFIDENCE_THRESHOLD = 0.85
UNKNOWN_THRESHOLD = 1.0
EXTREMELY_LOW_CONFIDENCE_THRESHOLD = 0.4

# Function to recognize faces
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
        if np.min(embedding_distances) > UNKNOWN_THRESHOLD * 2:  # Filter non-face embeddings
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

    return recognized_faces

# API endpoint for image processing
@api_app.post("/upload")
async def process_image(file: UploadFile = File(...)):
    # Read image file
    file_bytes = await file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv.imdecode(nparr, cv.IMREAD_COLOR)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    
    # Recognize faces
    recognized_faces = recognize_faces(image)

    # Return response
    response_data = {
        "recognized_faces": recognized_faces,
        "num_faces": len(recognized_faces),
        "threat_level": "Low" if all(name != "Unknown" for _, _, _, _, name in recognized_faces) else "High"
    }

    return JSONResponse(content=response_data)

# Run the API server with Uvicorn
if __name__ == "__main__":
    uvicorn.run(api_app, host="0.0.0.0", port=5000)
