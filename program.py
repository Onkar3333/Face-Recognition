
from tkinter import Frame
from turtle import right
from cv2 import VideoCapture
import face_recognition
import cv2          
import numpy as np
import csv
import os
from datetime import datetime

# pip install cmake
# pip install dlib
# pip install face_recognition

video_capture=cv2.VideoCapture(0);

onkar_image = face_recognition.load_image_file("photos/onkar.jpg")
onkar_encoading = face_recognition.face_encodings(onkar_image)[0]

shubham_image = face_recognition.load_image_file("photos/shubham.jpg")
shubham_encoading = face_recognition.face_encodings(shubham_image)[0]

Shrikrishna_image = face_recognition.load_image_file("photos/Shrikrishna.jpg")
Shrikrishna_encoading = face_recognition.face_encodings(Shrikrishna_image)[0]

amol_image = face_recognition.load_image_file("photos/amol.jpg")
amol_encoading = face_recognition.face_encodings(amol_image)[0]

aniket_image = face_recognition.load_image_file("photos/aniket.jpg")
aniket_encoading = face_recognition.face_encodings(aniket_image)[0]

tohid_image = face_recognition.load_image_file("photos/tohid.jpg")
tohid_encoading = face_recognition.face_encodings(tohid_image)[0]

magdum_image = face_recognition.load_image_file("photos/magdum.jpg")
magdum_encoading = face_recognition.face_encodings(magdum_image)[0]

sohan_image = face_recognition.load_image_file("photos/sohan.jpg")
sohan_encoading = face_recognition.face_encodings(sohan_image)[0]


known_face_encoding=[
    onkar_encoading,
    shubham_encoading,
    Shrikrishna_encoading,
    amol_encoading,
    aniket_encoading,
    tohid_encoading,
    magdum_encoading,
    sohan_encoading
]

known_face_names=[
    "Onkar Bansode",
    "Shubham Sawant",
    "Shrikrishna Sutar",
    "Amol Nadgire",
    "Aniket Shelake",
    "Tohid Patel",
    "Magdum Pirjade",
    "Sohan Basutkar"
]

students=known_face_names.copy()

face_locations=[]
face_encodings=[]
face_names=[]
s=True

now=datetime.now()
current_date=now.strftime("%d-%m-%Y")

f=open(current_date+'.csv','w+',newline='')
lnwriter=csv.writer(f)

while True:
    _,frame=video_capture.read()
    small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame=small_frame[:,:,::-1]
    if s:
        face_locations=face_recognition.face_locations(rgb_small_frame)
        face_encodings=face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names=[]

        for face_encoding, face_location in zip(face_encodings,face_locations):
            matches=face_recognition.compare_faces(known_face_encoding,face_encoding)
            name=""
            face_distance=face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index=np.argmin(face_distance)
            if matches[best_match_index]:
                name=known_face_names[best_match_index]

            face_names.append(name)
            if name in known_face_names:

                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                thickness = 2
                text_size = cv2.getTextSize(name, font, fontScale, thickness)[0]
                rect_top_left = (face_location[3] * 4, face_location[0] * 4 - text_size[1])
                rect_bottom_right = (face_location[1] * 4, face_location[2] * 4)
 
                cv2.rectangle(frame, rect_top_left, rect_bottom_right, (0, 255, 0), thickness)
                cv2.putText(frame, name, (rect_top_left[0] + 6, rect_bottom_right[1] - 6), font, fontScale, (255, 255, 255), thickness)

                if name in students:
                    students.remove(name)
                    print(name)
                    current_time=now.strftime("%H:%M:%S")
                    lnwriter.writerow([name,current_time])

    cv2.imshow("Attendence System",frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()
f.close()