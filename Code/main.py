import cv2 as cv
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'images'
images= []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images) :
    encodeList = []
    for img in images:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList

def markAttendance(name):
    with open('Code/Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList =[]
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
       
print('Encoding...')
encodeListKnown = findEncodings(images)
print('Encoding completed')

cap = cv.VideoCapture(0)

while True:
    success , img = cap.read()
    imgS = cv.resize(img,(0,0),None,0.25,0.25)
    imgS = cv.cvtColor(imgS, cv.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv.FILLED)
            cv.putText(img,name,(x1+6,y2-6),cv.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)


    cv.imshow('webcam', img)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

cap.release();
cv.destroyAllWindows();



