# import keras
import numpy as np
import cv2

video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not video_capture.isOpened():
    print('Could not open video device')

cam_width = 1280
cam_height = 720

video_capture.set(3,cam_width)
video_capture.set(4,cam_height)

# model = keras.models.load_model('optimized_model.h5')

while True:
    success, imgOriginal = video_capture.read()
    imgGray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)
    resized_array = cv2.resize(imgGray,(320,240),interpolation=cv2.INTER_AREA)
    resized_array = cv2.equalizeHist(resized_array)
    resized_array = resized_array/255
    image_feed_cnn = np.array(resized_array).reshape(-1, 240, 320, 1)

    #Predict using image feed
    # classIndex = int(model.predict_classes(image_feed_cnn))
    # predictions = model.predict(image_feed_cnn)
    # probability = np.amax(predictions)

    # if probability > 0.65:
    #     text = f'Classification:{str(classIndex)} , Probability:{str(probability)}'
    #     cv2.putText(resized_array,text,(50,50),cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
    
    cv2.imshow('image',imgOriginal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        video_capture.release()
        break

