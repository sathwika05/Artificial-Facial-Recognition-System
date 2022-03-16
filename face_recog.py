import face_recognition
import cv2
import os
import numpy as np
import datetime
from scipy import spatial


def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

#####################################################################################

def who_is_it(pred,employees):
    min_distance=0
    identity = "unknown"
    for (name,[idx,enc]) in employees.items():
        dist = 1 - spatial.distance.cosine(enc,pred)
        if dist > min_distance:
            min_distance=dist
            identity = name
            ids = idx
        if min_distance < 0.95:
            identity = "unknown"
            ids = 9999
    return min_distance,identity, ids

font = cv2.FONT_HERSHEY_SIMPLEX
cap =  cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 10.0, (640,480))


while True:
        ret,frame = cap.read()
        output = frame.copy()
        #try:
        if True:
            rgb = frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb)
            face_encodings = face_recognition.face_encodings(rgb, face_locations)
            for (face_location,face_encoding) in zip(face_locations,face_encodings):
                top, right, bottom, left = face_location
                face = frame[top:bottom, left:right]
                employees = np.load('embeddings_face.npy',allow_pickle=True).item()
                min_distance,identity, ids = who_is_it(face_encoding,employees)
                if identity != "unknown":
                        cv2.putText(output,"Emp ID: " + str(ids), (left-100, top), font, 0.8, (0,255,0), 2, cv2.LINE_AA)
                        cv2.putText(output,"Name: " + str(identity), (left-100, top-20), font, 0.8, (0,255,0), 2, cv2.LINE_AA)
                        cv2.putText(output,str(datetime.datetime.now()), (left-100, top-40), font, 0.8, (0,255,0), 2, cv2.LINE_AA)
                        draw_border(output, (left, top), (right, bottom), (255, 0, 105),2, 15, 10)
                else:
                    cv2.putText(output,"Name: " + "Unknown", (left-100, top), font, 0.8, (0,255,0), 2, cv2.LINE_AA)
                    draw_border(output, (left, top), (right, bottom), (255, 0, 105),2, 15, 10)
        #except:
            #continue

        frame = cv2.resize(output, (640,480))
        cv2.imshow("frame",frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
