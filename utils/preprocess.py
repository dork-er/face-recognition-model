import cv2 as cv

def preprocess_faces(image, haarcascade):
    """
    Detect and preprocess all faces in the image.
    Returns a list of tuples: (x, y, w, h, face_region).
    """
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)

    face_regions = []
    for (x, y, w, h) in faces:
        face_region = image[y:y+h, x:x+w]  # Crop the face region
        face_regions.append((x, y, w, h, face_region))

    return face_regions
