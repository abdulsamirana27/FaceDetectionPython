import cv2
import time
import imutils
import BoundingBox
import f_detector
from enums import face_rotation_enum
# defining face detector

# instanciar detector
detector = f_detector.detect_face_orientation() 

class VideoCamera(object):
    x = '1'
    resetCounter = 0
    rightCounter = 0
    leftCounter = 0
    startProcess = False
    star_time = None
    def __init__(self):
       #capturing video
       self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        #releasing camera
        self.video.release()
    def get_frame(self,face_rotation):
            #extracting framesx = '1'
            self.star_time = time.time()
            ret, frame = self.video.read()
            # frame = cv2.flip(frame, 1)
            frame = imutils.resize(frame,width=720)
            #-------------------------- Insertar preproceso -------------------------------------
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # detectar si hay un rostro frontal o de perfil
            boxes,names = detector.face_orientation(gray)
            if(len(names)>0):
                return self.detectFace(face_rotation,frame,boxes,names)
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()
            # print(detector.face_orientation(gray))

            # if(names[0]=='frontal'):
            #     resetCounter += 1
            #     if(resetCounter>10):
            #         resetCounter=0
            # if(len(names)>0):

            #     if(names[0]=='frontal'):
            #         self.startProcess = True

            #     if(self.startProcess):
            #         if(names[0]=='right'):
            #           self.rightCounter += 1
            #         if(self.rightCounter > 10):
            #             self.rightCounter = 0
            #             print("right scenario pass")

            #     if(self.startProcess):
            #         if(names[0]=='left'):
            #           self.leftCounter += 1
            #         if(self.leftCounter > 10):
            #             self.leftCounter = 0
            #             print("left scenario pass")
                 
    def detectFace(self,face_rotation,frame,boxes,names):
        print(face_rotation)
        if (face_rotation == names[0]):
            frame = BoundingBox.bounding_box(frame,boxes,names)
            # print(frame)
            # ----------------------------------------------------------------------------
            end_time = time.time() - self.star_time    
            FPS = 1/end_time
            cv2.putText(frame,f"FPS: {round(FPS,3)}",(10,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        elif (face_rotation == names[0]):
            frame = BoundingBox.bounding_box(frame,boxes,names)
            # print(frame)
            # ----------------------------------------------------------------------------
            end_time = time.time() - self.star_time    
            FPS = 1/end_time
            cv2.putText(frame,f"FPS: {round(FPS,3)}",(10,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        else:
            pass
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()