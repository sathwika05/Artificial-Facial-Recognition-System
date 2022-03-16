import cv2
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()        

    def get_frame(self):
        ret, frame = self.video.read()

        face_cascade = cv2.CascadeClassifier('face_read.xml') 
        
        # DO WHAT YOU WANT WITH TENSORFLOW / KERAS AND OPENCV
        # Detects faces of different sizes in the input image 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        faces = face_cascade.detectMultiScale(gray, 1.3,10)
        font = cv2.FONT_HERSHEY_SIMPLEX 
        
        lis = [0, 0, 0, 0]
        for (x,y,w,h) in faces: 
            # To draw a rectangle in a face  

            lis = [x, y, x+w, y+h]
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
            # cv2.putText(frame,  'TEXT ON VIDEO',  (50, 50),  font, 1,  (0, 255, 255),  2,  cv2.LINE_4) 

        ret, jpeg = cv2.imencode('.jpg', frame)

        return jpeg.tobytes(), frame, lis