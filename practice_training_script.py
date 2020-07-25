from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import cv2
import os
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf

image_paths = os.listdir('car_images')
full_image_paths = []
for image in image_paths:
    full_image_paths.append(f"car_images/{image}")

image_arrays = []
image_labels = []
for x in range(len(full_image_paths)):
    width = 240
    height = 320
    try:
        array_gray = cv2.imread(full_image_paths[x], cv2.IMREAD_GRAYSCALE)
        resized_array_gray = cv2.resize(array_gray,(height,width),interpolation=cv2.INTER_AREA)
        resize_array_rgb = cv2.cvtColor(resized_array_gray,cv2.COLOR_GRAY2RGB)
        image_arrays.append(resize_array_rgb)
        print(full_image_paths[x])
        label_split = image_paths[x].split('_')
        image_labels.append(label_split[0])
    except:
        pass

image_arrays = np.array(image_arrays).reshape(-1, 240, 320, 3)

label_encoder = LabelEncoder()
label_encoder.fit(image_labels)
encoded_y = label_encoder.transform(image_labels)

y_categorical = to_categorical(encoded_y)
classifications = len(y_categorical[0])

X_train, X_test, y_train, y_test = train_test_split(image_arrays,y_categorical,random_state=42)

dense_layers = [2]
layer_sizes = [32,64,128]
conv_layers = [1,2,3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = f"{conv_layer}-conv-{layer_size}-nodes-{dense_layer}-dense-{int(time.time())}"
            tensorboard = TensorBoard(log_dir=f'logs/{NAME}')
            print(NAME)

            model = Sequential()

            model.add(Conv2D(layer_size, (3, 3), activation='relu', input_shape=(240, 320, 3)))
            model.add(MaxPooling2D((2, 2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3), activation='relu'))
                model.add(MaxPooling2D((2, 2)))
            
            model.add(Flatten())

            for l in range(dense_layer):
                model.add(Dense(layer_size, activation='relu'))

            model.add(Dense(units=classifications,activation='softmax'))

            model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
            
            model.fit(X_train, y_train, epochs=20, 
                    validation_data=(X_test, y_test), callbacks=[tensorboard], verbose=2)

for conv_layer in convlayers:
    NAME = f"{conv_layer}-conv-128-nodes-0-dense-{int(time.time())}"
    tensorboard = TensorBoard(log_dir=f'logs/{NAME}')
    print(NAME)

    model = Sequential()

    model.add(Conv2D(128, (3, 3), activation='relu', input_shape=(240, 320, 3)))
    model.add(MaxPooling2D((2, 2)))

    for l in range(conv_layer-1):
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
    
    model.add(Flatten())

    model.add(Dense(units=classifications,activation='softmax'))

    model.compile(optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=20, 
            validation_data=(X_test, y_test), callbacks=[tensorboard], verbose=2)

for conv_layer in convlayers:
    NAME = f"{conv_layer}-conv-128-nodes-1-dense-{int(time.time())}"
    tensorboard = TensorBoard(log_dir=f'logs/{NAME}')
    print(NAME)

    model = Sequential()

    model.add(Conv2D(128, (3, 3), activation='relu', input_shape=(240, 320, 3)))
    model.add(MaxPooling2D((2, 2)))

    for l in range(conv_layer-1):
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
    
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(Dense(units=classifications,activation='softmax'))

    model.compile(optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=20, 
            validation_data=(X_test, y_test), callbacks=[tensorboard], verbose=2)
