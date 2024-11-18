import cv2 as cv

def preprocess_image(image, haarcascade):
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]  # Use the first detected face
    face_img = image[y:y+h, x:x+w]
    return face_img
