# Artificial-Facial-Recognition-System
An application to detect fake or proxy attendance using Python.
This code trains and implements liveness detection with video input.

Install steps:

* pip install imutils

* pip install keras

* pip install --upgrade h5py

* pip install opencv-python

Train & Deploy steps:
use ./gather.bat to get train image snapshots from video in ./videos folder. (images will be stored in ./dataset/fake & real)

run ./trainLiveness.bat to train images from ./dataset

run ./runLiveness.bat to see implementaion of Liveness Detection via video input (-c 0.5 is the threshold of "Fake" and "Liveness")

**Classification Report of the Model:**
<br>
<br>
![image](https://user-images.githubusercontent.com/55009076/158629960-beed50a3-c785-41d7-824b-59e113e35843.png)





 
  
 
 
  
  
 




