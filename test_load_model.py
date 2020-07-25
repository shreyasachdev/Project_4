import keras
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

image_paths = os.listdir('car_images')
full_image_paths = []
for image in image_paths:
    full_image_paths.append(f"car_images/{image}")

image_labels = []

for x in range(len(full_image_paths)):
    label_split = image_paths[x].split('_')
    image_labels.append(label_split[0])

label_encoder = LabelEncoder()
label_encoder.fit(image_labels)
encoded_y = label_encoder.transform(image_labels)

model = keras.models.load_model('optimized_model.h5')

test_picture_path = 'test_pictures/my20_civic_sedan_highlights_desktop_01.jpg'

array = cv2.imread(test_picture_path, cv2.IMREAD_GRAYSCALE)
resized_array = cv2.resize(array,(320,240),interpolation=cv2.INTER_AREA)
resized_array_rgb = cv2.cvtColor(resized_array,cv2.COLOR_GRAY2RGB)

image_array = np.array(resized_array_rgb).reshape(-1, 240, 320, 3)

classIndex = int(model.predict_classes(image_array))
predictions = model.predict(image_array)
probability = np.amax(predictions)
car_model_prediction = label_encoder.inverse_transform([classIndex])

print(f'Car Model: {car_model_prediction}, Prediction: {predictions}, Probability: {probability}')
