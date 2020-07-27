import keras
import numpy as np
import cv2
import joblib

label_encoder = joblib.load('label_encoders/label_encoder_42_classes.save')
video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not video_capture.isOpened():
    print('Could not open video device')

cam_width = 1280
cam_height = 720

video_capture.set(3,cam_width)
video_capture.set(4,cam_height)

model = keras.models.load_model('models/honda_volkswagen_model_1.h5')

while True:
    success, imgOriginal = video_capture.read()
    array = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)
    resized_array = cv2.resize(array,(320,240),interpolation=cv2.INTER_AREA)
    resized_array_rgb = cv2.cvtColor(resized_array,cv2.COLOR_GRAY2RGB)

    image_array = np.array(resized_array_rgb).reshape(-1, 240, 320, 3)

    try: 
        classIndex = int(model.predict_classes(image_array))
        predictions = model.predict(image_array)
        probability = np.amax(predictions)

        text = f'Classification:{label_encoder(str(classIndex))} , Probability:{str(probability)}'
        cv2.putText(imgOriginal,text,(50,50),cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
        
    except:
        pass

    cv2.imshow('image',imgOriginal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        video_capture.release()
        break


