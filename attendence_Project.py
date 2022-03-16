import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
path = os.getcwd() + "/images/"

import pandas as pd
import time 

# from gtts import gTTS 
# from playsound import playsound 

from ex import checkattendence, addnewrow, merge_file


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


from LBPFR import LBP

from skimage import feature
import numpy as np

from imutils import paths

from sklearn.svm import LinearSVC


desc = LBP.LocalBinaryPatterns(24, 8)
data = []
labels = []


# loop over the training images
for imagePath in paths.list_images("Training"):
	# load the image, convert it to grayscale, and describe it
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)
	# extract the label from the image path, then update the
	# label and data lists
	labels.append(imagePath.split(os.path.sep)[-2])
	data.append(hist)
# train a Linear SVM on the data
model = LinearSVC(C=100.0, random_state=42, max_iter=5000)
model.fit(data, labels)



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



images = []
classNames = []
mylist = os.listdir(path)
# print(mylist)
for cls in mylist:
    cur_Img = cv2.imread(f'{path}/{cls}')
    images.append(cur_Img)
    classNames.append(os.path.splitext(cls)[0])
print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img1)[0]
        encodeList.append(encode)
    return encodeList

Attendence_path_login = "Attendence_login.csv"
Attendence_path_logout = "Attendence_logout.csv"

checkattendence(Attendence_path_login)
checkattendence(Attendence_path_logout)


def return_primary(Attendence_path):

    data_path = pd.read_csv(Attendence_path)

    name_vals = data_path['namedate'].values
    date_vals = data_path['Date'].values
    
    return name_vals, date_vals

def markAttendance(name, Attendence_path_login, Attendence_path_logout):
    
    Date = ''
    Name = ''
    nameDate = ''
    LoginTime = ''
    LogOutTime = ''
    present_time_to_append = datetime.now().time()

    present_time_min_login = 0
    present_time_min_logout = 0

    present_time_hour_login = 0
    present_time_hour_logout = 0


    present_time_hour = datetime.now().time().hour
    present_time_min = datetime.now().time().minute

    # print(present_time_min)


    present_date = datetime.now().date()


    Names_list, Dates = return_primary(Attendence_path_login)
    Name_list_logout, _ = return_primary(Attendence_path_logout)

    nameDate = str(present_date) + name
    # print("Names_List", str(nameDate))


    time_in = 18
    present_min_in = 55
    if nameDate not in Names_list:
        if present_time_hour in [time_in] and present_time_min <= present_min_in:

            LoginTime = present_time_to_append
            Date = present_date
            Name = name
            nameDate = str(present_date) + name
            addnewrow(Attendence_path_login, Date=Date, Name=Name, nameDate=nameDate, LoginTime=LoginTime)
            present_time_min_login = present_time_min

    time_out = 18
    present_min_out = 56
    if nameDate in Names_list and present_time_hour in [time_out] and nameDate not in Name_list_logout:
        if present_time_hour in [time_out] and present_time_min >= present_min_out and str(present_date) in Dates:
            LogOutTime = present_time_to_append
            Name = name
            nameDate = str(present_date) + name
            addnewrow(Attendence_path_logout, Date=Date, Name=Name, nameDate=nameDate, LogOutTime=LogOutTime)
            present_time_min_logout = present_time_min



    csv_File_final = merge_file(Attendence_path_login, Attendence_path_logout)
    csv_File_final.to_csv("final.csv", index=False)

encodeList_Known = findEncodings(images)
# print('Encoding Completed')

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, int(20))



#Blink
EYE_AR_THRESH = 0.4
EYE_AR_CONSEC_FRAMES = 3

# print('[INFO] loading facial landmark predictor...')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

face_cascade = cv2.CascadeClassifier('face_read.xml') 


(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']



# cap = cv2.VideoCapture(0) Blink End


COUNTER = 0
TOTAL = 0


oldTotal = 0
# newTotal = 0

while True:

    

    Flag = False
    success, img = cap.read()


    frame = img.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)
    

    try:
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
                # print(f"This is the value for the {Flag}")
                TOTAL += 1
                # newTotal = TOTAL
                # print(f"This is the value for the {TOTAL}")
                Flag = True
                # print(f"This is the value for the {Flag}")

                

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
            'EYE: {:.2f}'.format(ear),
            (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0xFF),
            2,
            )
    except:
        print("[INFO] Passing the ErroR")


    faces = face_cascade.detectMultiScale(gray, 1.3,10)
    for (x,y,w,h) in faces: 
        # To draw a rectangle in a face  
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
        

        face_crop = gray[y:y+h, x:x+w]
        
        # print(face_crop.shape)

        # face_crop = cv2.resize(face_crop, (150, 150))
        # cv2.imshow("Face_Crop", face_crop)
        
        hist = desc.describe(face_crop)
        prediction = model.predict(hist.reshape(1, -1))
        # print(prediction)

        if prediction[0] == "Fake":
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 0, 255),2)

            # cv2.putText(frame, prediction[0], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        elif prediction[0] == "Real":
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 255, 0),2)

            # cv2.putText(frame, prediction[0], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        else:
            pass

    try:
        if oldTotal < TOTAL and prediction[0] == "Real" and Flag == True:
            imgs = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
            imgs1 = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
            faces_current_frame = face_recognition.face_locations(imgs1)
            encode = face_recognition.face_encodings(imgs1, faces_current_frame)

            for encode_face, face_loc in zip(encode, faces_current_frame):
                oldTotal = TOTAL 
                matches = face_recognition.compare_faces(encodeList_Known, encode_face)
                face_dis = face_recognition.face_distance(encodeList_Known, encode_face)
                # print(face_dis)
                # print(matches)
                matchindex = np.argmin(face_dis)

                if matches[matchindex] and face_dis[matchindex] <= 0.4:
                    print(f"Attendence Marked {prediction[0], classNames[matchindex]}")

                    name = classNames[matchindex].upper()
                    # print(name)
                    y1, x2, y2, x1 = face_loc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 0, 0), cv2.FILLED)
                    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 3)
                    Attendence_path_login = "Attendence_login.csv"
                    Attendence_path_logout = "Attendence_logout.csv"
                    markAttendance(name, Attendence_path_login, Attendence_path_logout)

                elif face_dis[matchindex] >= 0.5:

                    y1, x2, y2, x1 = face_loc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
                    cv2.putText(img, "UNKNOWN", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 3)
    except:
        print("[INFO] I'm passing the No Predictions")


    cv2.imshow('OutPut Video', img)
    cv2.imshow("Input Video", frame)
    if cv2.waitKey(25) & 0xff == ord("q"):
        break


cv2.destroyAllWindows()
cap.release()