from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2 as cv
import numpy as np
import pickle
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from utils.preprocess import preprocess_image

app = FastAPI()

# Load models
haarcascade = cv.CascadeClassifier("models/haarcascade_frontalface_default.xml")
facenet = FaceNet()
encoder = LabelEncoder()
svm_model = None

try:
    # Load SVM model
    svm_model = pickle.load(open("models/svm_model.pkl", 'rb'))
    # Assume encoder classes are stored in the same pickle file
    encoder.classes_ = svm_model.classes_
except Exception as e:
    print(f"Error loading model: {e}")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not svm_model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Read uploaded file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv.imdecode(nparr, cv.IMREAD_COLOR)

    # Detect face
    face_img = preprocess_image(image, haarcascade)
    if face_img is None:
        return JSONResponse({"message": "No face detected"}, status_code=400)

    # Get embedding
    face_img = cv.resize(face_img, (160, 160))
    face_img = np.expand_dims(face_img, axis=0)
    embedding = facenet.embeddings(face_img)

    # Predict using SVM
    prediction = svm_model.predict(embedding)
    predicted_name = encoder.inverse_transform(prediction)[0]

    return {"name": predicted_name}

@app.get("/")
def root():
    return {"message": "Face Recognition API is running"}
