from flask import Flask, render_template, Response, jsonify
from flask import request
from camera import VideoCamera
import cv2
import os
import sys
import face_recognition
import pandas as pd


from recog import RecognitionCamera



app = Flask(__name__)
app.config["CACHE_TYPE"] = "null"

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(APP_ROOT, 'Tested')

# app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config["TEMPLATES_AUTO_RELOAD"] = True


video_stream = VideoCamera()
recog_stream = RecognitionCamera()

global identity

@app.route('/')
def index():
    return render_template('index.html')


@app.route("/data")
def data():
    return render_template("data.html")

@app.route("/recognize")
def recognize():
    return render_template("recognize.html")

@app.route('/snap', methods=[ 'POST', 'GET'])
def snap():
    Flag = True
    while True:
        frame, original, lis = video_stream.get_frame()
        
        if not frame:
            break
        else:
            pass

        classes = []
        identity = request.form.get("name")
        ids = request.form.get("id")
        classes.append(identity)
        print(classes)
        if request.form.get("submitda"):
            print("entered")
            if sum(lis) > 0 and Flag:
                cv2.imwrite(f"./images/{identity}_{ids}.jpg", original[lis[1]:lis[3], lis[0]:lis[2]])
                Flag = False
            else:
                pass
        else:
            pass

    return render_template("data.html")


def gen(camera):
    count = 0
    while True:
        frame, original, _  = camera.get_frame()

        if not frame:
            break
        else:
            pass

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(video_stream),
                    mimetype='multipart/x-mixed-replace; boundary=frame')





def gen_recog(camera):
    count = 0

    ide = []
    idd = []

    while True:
        frame, original, identity, ids  = camera.get_frame()

        if not frame:
            break
        else:
            pass

            
        ide.append(identity)
        idd.append(ids)
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    data = {

            "Identity" : ide,
            "ids" : idd
    }

    dataframe = pd.DataFrame(data=data)
    dataframe.to_csv("DataRead.csv")



@app.route('/recog_feed', methods=['GET', 'POST'])
def recog_feed():
    return Response(gen_recog(recog_stream),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True,port="5000")

