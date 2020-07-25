from flask import Flask, jsonify, request
import keras
import cv2
import numpy as np

app = Flask(__name__)
model = keras.model.load_model('optimized_model.h5')

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

    predictions = model.predict(image_reshaped)
    probability = np.amax(predictions)
    return jsonify({'predictions':predictions, 'probability':probability})

if __name__ == "__main__":
    app.run(host=host, port=5000)