from flask import Response, Flask, render_template
from imutils.video import VideoStream
import HandTrackingModule as htm
import cv2
import math
import numpy as np
import threading
import time

outputFrame = None
lock = threading.Lock()

app = Flask(__name__)

vs = VideoStream(src = 0).start()
time.sleep(2.0)

@app.route("/")
def index():
	return render_template("index.html")

def volumeControl():
    ''' This function will analyze each frame to detect the hands and
    the finger positions, calculate the distance between index and thumb, 
    convert it to a volume level, and annotate the frame with the relevant information.
    '''
    
    global vs, outputFrame, lock
    detector = htm.handDetector(detectionCon = 0.7)
    
    while True:
        frame = vs.read()
        frame = cv2.resize(frame, (640, 480))
        detector.findHands(frame, draw = False)
        lmList = detector.findPosition(frame, draw = False)[0]
        cv2.rectangle(frame, (50, 150), (85, 400), (0, 0, 255), 3)
        
        if len(lmList) != 0:
            cv2.putText(frame, 'MAO DETECTADA', (200, 450), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 3)
            x1, y1 = lmList[4][1:]
            x2, y2 = lmList[8][1:]	
            length = math.hypot(x2 - x1, y2 - y1)
            height = np.interp(length, [25, 200], [400, 150])
            perc = np.interp(length, [25, 200], [0, 100])
            cv2.rectangle(frame, (50, int(height)), (85, 400), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, f'{int(perc)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
            
        with lock:
            outputFrame = frame.copy()

def generate():
    ''' This function converts the frame to html output '''
    
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
			
            if not flag:
                continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route("/volume_control")
def video_feed():
	return Response(generate(), mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    t = threading.Thread(target = volumeControl)
    t.daemon = True
    t.start()
    app.run(host='0.0.0.0', debug = True, threaded = True, use_reloader = False)

vs.stop()