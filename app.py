from flask import Flask, jsonify, request
from tensorflow import keras
from cv2 import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

def preprocess_model ():

    image_labels = []
    for x in range(len(full_image_paths)):
        label_split = image_paths[x].split('_')
        image_labels.append(label_split[0])

    label_encoder = LabelEncoder()
    label_encoder.fit(image_labels)
    encoded_y = label_encoder.transform(image_labels)
    model = keras.models.load_model('optimized_model_2.h5')
    print("Model Loaded!")

preprocess_model ()

@app.route('/hello-world', methods=['GET', 'POST'])
def say_hello():
    return jsonify({'result': 'Hello world'})

@app.route('/predict',methods=['GET','POST'])
def predict():
    data = request.json
    array = cv2.imread(data, cv2.IMREAD_GRAYSCALE)
    resized_array = cv2.resize(array,(height,width),interpolation=cv2.INTER_AREA)
    resized_array_rgb = cv2.cvtColor(resized_array,cv2.COLOR_GRAY2RGB)
    image_reshaped = np.array(resized_array_rgb).reshape(-1, 240, 320, 3)

    classIndex = int(model.predict_classes(image_reshaped))
    predictions = model.predict(image_reshaped)
    probability = np.amax(predictions)
    car_model_prediction = label_encoder.inverse_transform([classIndex])
    return jsonify({'predictions':car_model_prediction, 'probability':probability})

if __name__ == "__main__":
    app.run(debug=True)