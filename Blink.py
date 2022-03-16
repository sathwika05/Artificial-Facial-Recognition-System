from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2


def eye_aspect_ratio(eye):

    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates

    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates

    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio

    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio

    return ear




EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3



COUNTER = 0
TOTAL = 0



print('[INFO] loading facial landmark predictor...')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')



(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']



cap = cv2.VideoCapture(0)


while True:

    
    _, frame = cap.read()
    

    print(frame)
    # frame = cv2.resize(frame, (450, 450), cv2.INTER_LINEAR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame

    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

        

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

        

    ear = (leftEAR + rightEAR) / 2.0

        

    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(frame, [leftEyeHull], -1, (0, 0xFF, 0), 1)
    cv2.drawContours(frame, [rightEyeHull], -1, (0, 0xFF, 0), 1)

        

    if ear < EYE_AR_THRESH:
        COUNTER += 1
    else:



        if COUNTER >= EYE_AR_CONSEC_FRAMES:
            TOTAL += 1

            

        COUNTER = 0

        

    cv2.putText(
        frame,
        'Blinks: {}'.format(TOTAL),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0xFF),
        2,
        )
    cv2.putText(
        frame,
        'EAR: {:.2f}'.format(ear),
        (300, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0xFF),
        2,
        )

    # show the frame

    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop

    if key == ord('q'):
        break

# do a bit of cleanup

cv2.destroyAllWindows()
cap.release()