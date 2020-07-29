from tensorflow import keras
import joblib
import numpy as np
import cv2

def Test_model():
    cnn_path = 'optimized_model.h5'
    model = keras.models.load_model(cnn_path)
    label_encoder = joblib.load("label_encoder_42_classes.save")
    test_picture_url = 'IMAGE.png'
    array = cv2.imread(test_picture_url, cv2.IMREAD_GRAYSCALE)
    resized_array = cv2.resize(array,(320,240),interpolation=cv2.INTER_AREA)
    resized_array_rgb = cv2.cvtColor(resized_array,cv2.COLOR_GRAY2RGB)
    image_array = np.array(resized_array_rgb).reshape(-1, 240, 320, 3)
    number = model.predict_classes(image_array)
    return ((label_encoder.inverse_transform(number))[0])

