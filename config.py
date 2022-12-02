detect_frontal_face = 'haarcascades/haarcascade_frontalface_alt.xml'
detect_perfil_face = 'haarcascades/haarcascade_profileface.xml'
EYE_AR_THRESH = 0.23 #baseline
EYE_AR_CONSEC_FRAMES = 3

# eye landmarks
eye_landmarks = "model_landmarks/shape_predictor_68_face_landmarks.dat"
# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

# face_model for emotion
detect_frontal_face = 'haarcascades/haarcascade_frontalface_alt.xml'
# model path
path_model = './emotion_model/model_dropout.hdf5'
w,h = 48,48
rgb = False
labels = ['angry','disgust','fear','happy','neutral','sad','surprise']