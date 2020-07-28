from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import cv2
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
from keras.optimizers import Adam
import joblib

#Get all image paths from car_images folder and append those paths to a list
image_paths = os.listdir('5_car_brands')
full_image_paths = []
for image in image_paths:
    full_image_paths.append(f"5_car_brands/{image}")

#Create empty lists, one for the image arrays, and another for the car brand label
image_arrays = []
image_labels = []

for x in range(len(full_image_paths)):
    width = 240
    height = 320
    try:
        #Loop through each full image path and load each image as grayscale using cv2
        array = cv2.imread(full_image_paths[x], cv2.IMREAD_GRAYSCALE)

        #Resize each array specified width and height
        resized_array = cv2.resize(array,(height,width),interpolation=cv2.INTER_AREA)

        #Convert back to rgb to add color channel back to array shape (Image still remains in grayscale)
        resized_array_rgb = cv2.cvtColor(resized_array,cv2.COLOR_GRAY2RGB)
        #Append converted array to image_arrays list
        image_arrays.append(resized_array_rgb)
        print(full_image_paths[x])

        #Simultaneously get the images brand from the image file name
        label_split = image_paths[x].split('_')
        image_labels.append(label_split[0])
    except Exception as e:
        print('failed')

#Resize array to fullfill CNN required shape
image_arrays = np.array(image_arrays).reshape(-1, 240, 320, 3)

#Use label encoder and to_categorical to one hot encode the car brand labels
label_encoder = LabelEncoder()
label_encoder.fit(image_labels)
encoded_y = label_encoder.transform(image_labels)
y_categorical = to_categorical(encoded_y)

#Get Number of car brand classifications (needed for dense layer output in CNN)
classifications = len(y_categorical[0])

#Save label_encoder object so that predictions can be transformed in other jupyter notebooks/python scripts
label_encoder_filename = "label_encoders/5_car_brands.save"
joblib.dump(label_encoder, label_encoder_filename) 

#Split data set into train and test sets. Then split set again to get validation set
X_train, X_test, y_train, y_test = train_test_split(image_arrays,y_categorical,random_state=42)
X_train, X_val, y_train, y_val = train_test_split(image_arrays,y_categorical,test_size=0.1,random_state=42)

#Define parameters for model
sizeOfFilter1 = (5,5)
sizeOfFilter2 = (3,3)
opt = Adam(lr=0.03)

#Initialize tensorboard object to be used for callbacks
NAME = 'Optimized_Model_5_classes'
print(NAME)
tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

#Initalize model, and add layers
model = Sequential()

model.add(Conv2D(128,sizeOfFilter1,activation='relu',input_shape=(240,320,3)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(256,sizeOfFilter2,activation='relu'))
model.add(Conv2D(256,sizeOfFilter2,activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(256,sizeOfFilter2,activation='relu'))
model.add(Conv2D(256,sizeOfFilter2,activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(units=classifications,activation='softmax'))

#Compile model
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#Fit model 
try:
    model.fit(X_train, y_train,
                epochs=10,
                batch_size=32,
                shuffle=1,
                callbacks=[tensorboard])
except Exception as e:
    print('Error: %s', e)

print(model.evaluate(X_test, y_test))

#Save model
model.save('models/5_car_brands_model.h5')