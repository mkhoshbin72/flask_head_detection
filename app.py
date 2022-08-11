from flask import  Flask, render_template, Response
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import StringField, PasswordField, SubmitField, BooleanField, TextAreaField, RadioField

from detect import detect, load_model

import cv2


app = Flask(__name__)
app._static_folder = 'static'

# cap = cv2.VideoCapture('/home/mkh/Downloads/out.mp4')
cap = cv2.VideoCapture(0)


def get_frame():

    frame_counter = 0
    weight = ''
    device = 'cpu'
    model = load_model(weight=weight, device=device)

    while True:
        ret, frame = cap.read()

        frame_counter += 1
        # if you run the code on video and want to replay video after it finished uncomment the code bellow
        # if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        #     frame_counter = 0
        #     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if not ret:
            break
        if frame_counter % 10 == 0:
            frame = detect(frame, model)
            ret_, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)