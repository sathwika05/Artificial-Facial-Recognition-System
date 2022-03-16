from scipy.spatial.distance import _filter_deprecated_kwargs
import face_recognition
import cv2
import os
import numpy as np




#Capturing the image

def Image(img, nameid="", name=""):
    
    face_cascade = cv2.CascadeClassifier('face_read.xml') 
    
    # reads frames from a camera 
    # img = cv2.imread(img)  
      
    # convert to gray scale of each frames 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
      
    # Detects faces of different sizes in the input image 
    faces = face_cascade.detectMultiScale(gray, 1.3,10)
      
    for (x,y,w,h) in faces: 
        # To draw a rectangle in a face  
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)  
            
        #Saving the detected face in the Live Stream
        cv2.imwrite(f"{nameid}/{name}.jpg", img[y:y+h, x:x+w])
          
    # Display an image in a window 
    cv2.imshow('img',img) 
    cv2.waitKey(0)
    # De-allocate any associated memory usage 
    cv2.destroyAllWindows()


def Captureimage(webcam=0, nameid="", name=""):

    capture = cv2.VideoCapture(webcam)

    i =0
    while i < 3:

        ret, frame = capture.read()

        cv2.imshow("frame", frame)
        Image(frame, nameid=nameid, name=name)
        i += 1
    cv2.waitKey()
    cv2.destroyAllWindows()
    capture.release()



def create_emidname(name):

    path, dirs, files = next(os.walk(os.getcwd()+"/images/"))
    file_count = len(files) + 1

    name_id = name + "_" + str(file_count)

    if os.path.exists(os.getcwd()+"/data/"+name_id):
        pass
    else:
        path = os.getcwd()+"/data/"+name_id
        os.mkdir(path)
        Captureimage(nameid=path, name=name)



def train():
    data_dict={}
    path="data/"
    total = 0
    for persons in os.listdir(path):
        pname = path + persons
        for images in os.listdir(pname):
            image_path = pname + '/' + images
            name=persons.split("_")[0]
            idx = persons.split("_")[1]
            image = cv2.imread(image_path)
            print(image.shape)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb)
            encodings = face_recognition.face_encodings(image, boxes)
            for encoding in encodings:
                data_dict[str(name)]=data_dict.get(name,[idx, encoding])
                np.save("embeddings_face.npy",data_dict)
                print(f"[info] Saved Encodings {total}")
        total +=1 


def train_data(data):
    if data.lower() == "new":
        numberofpeoples = int(input("Please enter the number Persons You are interested to Capture :- "))
        for i in range(numberofpeoples):
            username = input("Please the the name of the User :- ")
            create_emidname(name=username)
            train()
            
    elif data.lower() == "old":
        train()
    else:
        print("[Info] please enter old or new")


if __name__ == "__main__":
    data = input("Enter the new data :- ")
    train_data(data)