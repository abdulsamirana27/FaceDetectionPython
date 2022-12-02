from flask import Flask, render_template, Response,request,redirect
from camera import VideoCamera
face_rotation = ""
app = Flask(__name__)
@app.route('/',methods= ['POST','GET',])
def index():
    return faceDetection()
     
def gen(camera,face_rotation):
    while True:
        #get camera frame
        frame = camera.get_frame(face_rotation)
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
def faceDetection():
    if request.form.get('faceleft')!=None:
        face_rotation="left"
    elif  request.form.get('faceright')!=None:
        face_rotation="right"
    elif request.form.get('faceblink')!=None:
        face_rotation="blink"
    elif request.form.get('facesmiley')!=None:
        face_rotation="smile"
    else:
        face_rotation="frontal"
    return render_template("menu.html",face_rotation=face_rotation)

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera(),request.args["face_rotation"]),mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == '__main__':
    # defining server ip address and port
    app.run(host='0.0.0.0',port='5000', debug=True)