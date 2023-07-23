import mtcnn

face_detector = mtcnn.MTCNN()
print(help(face_detector.detect_faces))

image = cv2.imread('images/face.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# disabled mtcnn verbose messages
