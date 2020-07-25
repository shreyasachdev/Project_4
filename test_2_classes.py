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

image_paths = os.listdir('mazda_jeep_images')
full_image_paths = []
for image in image_paths:
    full_image_paths.append(f"mazda_jeep_images/{image}")

image_arrays = []
image_labels = []

for x in range(len(full_image_paths)):
    width = 240
    height = 320
    try:
        array = cv2.imread(full_image_paths[x], cv2.IMREAD_GRAYSCALE)
        resized_array = cv2.resize(array,(height,width),interpolation=cv2.INTER_AREA)
        resized_array_rgb = cv2.cvtColor(resized_array,cv2.COLOR_GRAY2RGB)
        image_arrays.append(resized_array_rgb)
        print(full_image_paths[x])
        label_split = image_paths[x].split('_')
        image_labels.append(label_split[0])
    except Exception as e:
        print('failed')

image_arrays = np.array(image_arrays).reshape(-1, 240, 320, 3)

label_encoder = LabelEncoder()
label_encoder.fit(image_labels)
label_encoder_filename = "label_encoders/label_encoder_mazda_jeep.save"
joblib.dump(label_encoder, label_encoder_filename) 
encoded_y = label_encoder.transform(image_labels)

y_categorical = to_categorical(encoded_y)
classifications = len(y_categorical[0])

X_train, X_test, y_train, y_test = train_test_split(image_arrays,y_categorical,random_state=42)

# dataGen = ImageDataGenerator(width_shift_range=0.1,
#                             height_shift_range=0.1,
#                             zoom_range=0.2,
#                             shear_range=0.1,
#                             rotation_range=10)
# dataGen.fit(X_train)

noOfFilters1 = 64
noOfFilters2 = 32
sizeOfFilter1 = (5,5)
sizeOfFilter2 = (3,3)
opt = Adam(lr=0.03)

NAME = 'Optimized_Model'
print(NAME)
tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

model = Sequential()

model.add(Conv2D(noOfFilters1,sizeOfFilter1,activation='relu',input_shape=(240,320,3)))
model.add(Conv2D(noOfFilters1,sizeOfFilter1,activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(noOfFilters2,sizeOfFilter2,activation='relu'))
model.add(Conv2D(noOfFilters2,sizeOfFilter2,activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(noOfFilters2,sizeOfFilter2,activation='relu'))
model.add(Conv2D(noOfFilters2,sizeOfFilter2,activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(units=classifications,activation='softmax'))

model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
try:
    model.fit(X_train, y_train,
                        epochs=20,
                        batch_size=64,
                        validation_split=0.1,
                        shuffle=1,
                        callbacks=[tensorboard])
except Exception as e:
    print('Error: %s', e)

print(model.evaluate(X_test, y_test))

model.save('models/mazda_jeep_model_3.h5')