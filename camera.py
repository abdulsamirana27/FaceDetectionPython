import cv2
import time
import imutils
import BoundingBox
import f_detector

# defining face detector

# instanciar detector
detector = f_detector.detect_face_orientation() 

class VideoCamera(object):
    x = '1'
    resetCounter = 0
    rightCounter = 0
    leftCounter = 0
    startProcess = False
    def __init__(self):
       #capturing video
       self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        #releasing camera
        self.video.release()
    def get_frame(self):
            #extracting framesx = '1'
           
            star_time = time.time()
            ret, frame = self.video.read()
            # frame = cv2.flip(frame, 1)
            frame = imutils.resize(frame,width=720)
            #-------------------------- Insertar preproceso -------------------------------------
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # detectar si hay un rostro frontal o de perfil
            boxes,names = detector.face_orientation(gray)
            # print(detector.face_orientation(gray))

            # if(names[0]=='frontal'):
            #     resetCounter += 1
            #     if(resetCounter>10):
            #         resetCounter=0
            if(len(names)>0):

                if(names[0]=='frontal'):
                    self.startProcess = True

                if(self.startProcess):
                    if(names[0]=='right'):
                      self.rightCounter += 1
                    if(self.rightCounter > 10):
                        self.rightCounter = 0
                        print("right scenario pass")

                if(self.startProcess):
                    if(names[0]=='left'):
                      self.leftCounter += 1
                    if(self.leftCounter > 10):
                        self.leftCounter = 0
                        print("left scenario pass")
                 
          
            frame = BoundingBox.bounding_box(frame,boxes,names)
            # print(frame)
            # ----------------------------------------------------------------------------
            end_time = time.time() - star_time    
            FPS = 1/end_time
            cv2.putText(frame,f"FPS: {round(FPS,3)}",(10,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            # cv2.imshow('preview',frame)
            # if cv2.waitKey(1) &0xFF == ord('q'):
            #     break
            # encode OpenCV raw frame to jpg and displaying it
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()