import config as cfg
import dlib
import cv2
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist


def detect(img, cascade):
    rects,_,confidence = cascade.detectMultiScale3(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                    flags=cv2.CASCADE_SCALE_IMAGE, outputRejectLevels = True)
    #rects = cascade.detectMultiScale(img,minNeighbors=10, scaleFactor=1.05)
    if len(rects) == 0:
        return (),()
    rects[:,2:] += rects[:,:2]
    return rects,confidence


def convert_rightbox(img,box_right):
    res = np.array([])
    _,x_max = img.shape
    for box_ in box_right:
        box = np.copy(box_)
        box[0] = x_max-box_[2]
        box[2] = x_max-box_[0]
        if res.size == 0:
            res = np.expand_dims(box,axis=0)
        else:
            res = np.vstack((res,box))
    return res


class detect_face_orientation():
    def __init__(self):
        # crear el detector de rostros frontal
        self.detect_frontal_face = cv2.CascadeClassifier(cfg.detect_frontal_face)
        # crear el detector de perfil rostros
        self.detect_perfil_face = cv2.CascadeClassifier(cfg.detect_perfil_face)
    def face_orientation(self,gray):
        # frontal_face
        box_frontal,w_frontal = detect(gray,self.detect_frontal_face)
        if len(box_frontal)==0:
            box_frontal = []
            name_frontal = []
        else:
            name_frontal = len(box_frontal)*["frontal"]
        # left_face
        box_left, w_left = detect(gray,self.detect_perfil_face)
        if len(box_left)==0:
            box_left = []
            name_left = []
        else:
            name_left = len(box_left)*["left"]
        # right_face
        gray_flipped = cv2.flip(gray, 1)
        box_right, w_right = detect(gray_flipped,self.detect_perfil_face)
        if len(box_right)==0:
            box_right = []
            name_right = []
        else:
            box_right = convert_rightbox(gray,box_right)
            name_right = len(box_right)*["right"]

        boxes = list(box_frontal)+list(box_left)+list(box_right)
        names = list(name_frontal)+list(name_left)+list(name_right)
        return boxes, names

class eye_blink_detector():
    def __init__(self):
        # cargar modelo para detecction frontal de rostros
        self.detector_faces = dlib.get_frontal_face_detector()
        # cargar modelo para deteccion de puntos de ojos
        self.predictor_eyes = dlib.shape_predictor(cfg.eye_landmarks)

    def eye_blink(self,gray,rect,COUNTER,TOTAL):
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = self.predictor_eyes(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = self.eye_aspect_ratio(leftEye)
        rightEAR = self.eye_aspect_ratio(rightEye)
        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0
        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < cfg.EYE_AR_THRESH:
            COUNTER += 1
        # otherwise, the eye aspect ratio is not below the blink
        # threshold
        else:
            # if the eyes were closed for a sufficient number of
            # then increment the total number of blinks
            if COUNTER >= cfg.EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
            # reset the eye frame counter
            COUNTER = 0
        return COUNTER,TOTAL

    def eye_aspect_ratio(self,eye):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])
        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        # return the eye aspect ratio
        return ear





def convert_rectangles2array(rectangles,image):
    res = np.array([])
    for box in rectangles:
        [x0,y0,x1,y1] = max(0, box.left()), max(0, box.top()), min(box.right(), image.shape[1]), min(box.bottom(), image.shape[0])
        new_box = np.array([x0,y0,x1,y1])
        if res.size == 0:
            res = np.expand_dims(new_box,axis=0)
        else:
            res = np.vstack((res,new_box))
    return res

def get_areas(boxes):
    areas = []
    for box in boxes:
        x0,y0,x1,y1 = box
        area = (y1-y0)*(x1-x0)
        areas.append(area)
    return areas

def bounding_box(img,box,match_name=[]):
    for i in np.arange(len(box)):
        x0,y0,x1,y1 = box[i]
        img = cv2.rectangle(img,
                    (x0,y0),
                    (x1,y1),
                    (0,255,0),3);
        if not match_name:
            continue
        else:
            cv2.putText(img, match_name[i], (x0, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    return img