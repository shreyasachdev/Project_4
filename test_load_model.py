import keras
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.externals import joblib

label_encoder = joblib.load('label_encoders/label_encoder_honda_volkswagen.save')
model = keras.models.load_model('models/honda_volkswagen_model_1.h5')

test_picture_path_1 = 'test_pictures/my20_civic_sedan_highlights_desktop_01.jpg'
test_picture_path_2 = 'test_pictures/honda_crv_2019_test.jpg'
test_picture_path_3 = 'test_pictures/volkswagen_golf.jpg'


array = cv2.imread(test_picture_path_3, cv2.IMREAD_GRAYSCALE)
resized_array = cv2.resize(array,(320,240),interpolation=cv2.INTER_AREA)
resized_array_rgb = cv2.cvtColor(resized_array,cv2.COLOR_GRAY2RGB)

image_array = np.array(resized_array_rgb).reshape(-1, 240, 320, 3)

classIndex = int(model.predict_classes(image_array))
predictions = model.predict(image_array)
probability = np.amax(predictions)
car_model_prediction = label_encoder.inverse_transform([classIndex])

print(f'Car Model: {car_model_prediction}, Prediction: {predictions}, Probability: {probability}')
