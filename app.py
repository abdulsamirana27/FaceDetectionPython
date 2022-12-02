# main.py
# import the necessary packages
from flask import Flask, render_template, Response,request
from camera import VideoCamera
face_rotation = "left"
app = Flask(__name__)
@app.route('/')
def index():
    author = True
    return render_template('menu.html',author=author)
     
def gen(camera):
    while True:
        #get camera frame
        frame = camera.get_frame('left')
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
@app.route('/face_detection')
def faceDetection():
    print(request.form.get('action1'))
    print(request.form.get('action2'))
    if request.method == 'POST':
        if request.form.get('faceleft') == 'faceleft':
            face_rotation="left"
        elif  request.form.get('faceright') == 'faceright':
            face_rotation="right"
        else:
            face_rotation="frontal" 
    return render_template("/")

@app.route('/video_feed')
def video_feed():
    args = request.args
    print(args)
    return Response(gen(VideoCamera()),mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == '__main__':
    # defining server ip address and port
    app.run(host='0.0.0.0',port='5000', debug=True)