from flask import request, Flask, render_template, Response
import cv2
import re
import os
import cvzone
import matplotlib.pyplot as plt
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import numpy
from PIL import Image


import numpy as np

app = Flask(__name__)

camera = cv2.VideoCapture(0)


def load_image(f1):
    # Read File Storage Files like Images taking from pc
    npimg = numpy.fromstring(f1, numpy.uint8)

    # Decode this file to read
    img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)

    # Resize this file into frame size
    img = cv2.resize(img, (640, 480))

    # For create a frame for removing background
    segmentor = SelfiSegmentation()

    while True:

        ## read the camera frame
        success, frame = camera.read()

        # Remove background taking first param - frame per images and img for background,threshold for sharpness
        imgout = segmentor.removeBG(frame, img, threshold=0.76)

        if not success:
            break
        else:
            # Now encode image
            ret, buffer = cv2.imencode('.jpg', imgout)

            # Convert it into bytes

            frame = buffer.tobytes()

            # Frames pass into this ,yield is because ,we need frame continusly ,if i used return it will display a single image
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def correct():
    return render_template('div - Copy.html')


@app.route('/success', methods=['POST'])
def success():
    # taking data and sending data to server
    if request.method == 'POST':
        f = request.files['file'].read()
        r = load_image(f)

    # Response will display
    return Response(r, mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run()




