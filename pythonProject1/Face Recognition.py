import cv2
import numpy as np
import face_recognition

img_train = face_recognition.load_image_file('image/S3.jpg')
img_train = cv2.cvtColor(img_train, cv2.COLOR_BGR2RGB)
img_test = face_recognition.load_image_file('Img/2100.png')
img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(img_train)[0]
encodeimg = face_recognition.face_encodings(img_train)[0]
cv2.rectangle(img_train, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (50, 0, 50), 2)

faceLocTest = face_recognition.face_locations(img_test)[0]
encodetest = face_recognition.face_encodings(img_test)[0]
cv2.rectangle(img_test, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (250, 0, 250), 2)

results = face_recognition.compare_faces([encodeimg], encodetest)
faceDis = face_recognition.face_distance([encodeimg], encodetest)
print(results, faceDis)
cv2.putText(img_test, f'{results}{round(faceDis[0],2)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('img_train', img_train)
cv2.imshow('img_test', img_test)

cv2.waitKey(0)
