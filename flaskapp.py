from flask import Flask, render_template, Response, jsonify, request, session
import os
import cv2
from YOLO_Video import video_detection

app = Flask(__name__)

app.config['SECRET_KEY'] = 'thesis'

def generate_frames(path_x=''):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_frames_web(path_x):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    session.clear()
    return render_template('landing.html')

@app.route("/webcam", methods=['GET', 'POST']) #for ip cam
def webcam():
    session.clear()
    return render_template('ui.html')

@app.route('/FrontPage', methods=['GET', 'POST'])
def front():
    # This route now simply renders the template without handling file upload.
    return render_template('videoprojectnew.html')

@app.route('/video')
def video():
    # Example path, adjust based on actual usage
    return Response(generate_frames(path_x='../Videos/try.mp4'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webapp') #for ipcam
def webapp():
    # Example usage, adjust the `path_x` parameter as needed
    return Response(generate_frames_web(path_x=1), mimetype='multipart/x-mixed-replace; boundary=frame')
    #return Response(generate_frames_web(path_x='rtsp://Dexios:Dexthegreat30@192.168.254.102:554/stream1'),mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == "__main__":
    app.run(debug=True)
