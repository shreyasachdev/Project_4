import keras
import numpy as np
import cv2
import joblib

#Initialize video capture from opencv
video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not video_capture.isOpened():
    print('Could not open video device')

#Set cam width and height for video capture
cam_width = 1280
cam_height = 720

video_capture.set(3,cam_width)
video_capture.set(4,cam_height)

#Load model 
model = keras.models.load_model('models/mazda_jeep_model_4.h5')

#Load associating label encoder using joblib
label_encoder = joblib.load('label_encoders/label_encoder_mazda_jeep.save')

while True:
    #read video image frame
    success, imgOriginal = video_capture.read()

    #Convert image to grayscale
    array = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)

    #Resize each array specified width and height
    resized_array = cv2.resize(array,(320,240),interpolation=cv2.INTER_AREA)

    #Convert back to rgb to add color channel back to array shape (Image still remains in grayscale)
    resized_array_rgb = cv2.cvtColor(resized_array,cv2.COLOR_GRAY2RGB)

    #Resize array to fullfill CNN required shape
    image_array = np.array(resized_array_rgb).reshape(-1, 240, 320, 3)

    try: 
        #Feed converted video image frame to loaded model and output predictions
        classIndex = np.argmax(model.predict(image_array), axis=-1)
        predictions = model.predict(image_array)
        probability = np.amax(predictions)

        #If probability greater than 50%
        if probability > 0.5:
            #Add classification and probability to original image feed from webcam
            text = f'Classification:{label_encoder.inverse_transform(classIndex)} , Probability:{str(probability)}'
            cv2.putText(imgOriginal,text,(50,50),cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
        
    except:
        pass
    
    #Show image in window
    cv2.imshow('image',imgOriginal)

    #Have image window open until user presses q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        video_capture.release()
        break


