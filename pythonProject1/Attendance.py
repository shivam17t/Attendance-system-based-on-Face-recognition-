import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
path = 'Attendance_File'
images = []
names = []
mylist = os.listdir(path)
print(mylist)

for i in mylist:
    Img = cv2.imread(f'{path}/{i}')
    images.append(Img)
    names.append(os.path.splitext(i)[0])
print(names)


def findencoding(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        namelist = []
        myData = f.readlines()
        for line in myData:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dt = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dt}')


encodeListKnown = findencoding(images)
print('Encoding Done')

cam = cv2.VideoCapture(0)

while True:
    success, img = cam.read()
    # reduce the size of image 1/4th of the size
    imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    faceLoc_img = face_recognition.face_locations(imgs)
    en_curr_img = face_recognition.face_encodings(imgs, faceLoc_img)
    
    for encodeFace,faceLoc in zip(en_curr_img, faceLoc_img):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchindex =np.argmin(faceDis)
        if matches[matchindex]:
            name = names[matchindex].upper()
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 255), cv2.FILLED)
            cv2.putText(img, name,(x1+6,y2-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)