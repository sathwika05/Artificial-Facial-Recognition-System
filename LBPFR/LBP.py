# import the necessary packages
from skimage import feature
import numpy as np



class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius
 
    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.numPoints + 3), range=(0, self.numPoints + 2))
 
		# normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
 
		# return the histogram of Local Binary Patterns
        return hist


desc = LocalBinaryPatterns(24, 8)
data = []
labels = []


from imutils import paths
import cv2
import os

from sklearn.svm import LinearSVC

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


for imagePath in paths.list_images("Testing"):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(gray.shape)
    hist = desc.describe(gray)
    prediction = model.predict(hist.reshape(1, -1))
    cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    print(prediction)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('face_read.xml') 

while cap.isOpened():

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3,10)

    for (x,y,w,h) in faces: 
        # To draw a rectangle in a face  
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
        face_crop = gray[y:y+h, x:x+w]
        
        print(face_crop.shape)

        # face_crop = cv2.resize(face_crop, (150, 150))
        cv2.imshow("Face_Crop", face_crop)
        
        hist = desc.describe(face_crop)
        prediction = model.predict(hist.reshape(1, -1))
        print(prediction)

        if prediction[0] == "Fake":
            cv2.putText(frame, prediction[0], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        elif prediction[0] == "Real":
            cv2.putText(frame, prediction[0], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        else:
            pass

    cv2.imshow("image", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
