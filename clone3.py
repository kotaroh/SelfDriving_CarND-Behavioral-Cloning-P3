import csv
import cv2
import numpy as np

images = []
measurements = []
augmented_images, augmented_measurements = [],[]

current_paths = ['./Data2/test/','./data/','./Data2/Test5/', './Data2/Test7/','./Data2/Test8/','./Data2/Test9/','./Data2/Test10/','./Data2/Test11/','./Data2/Test12/','./Data2/Test2/']

#'./Data2/Test3/','./Data2/Test4/'

for current_path in current_paths:
    lines = []
    csv_path = current_path + 'driving_log.csv'
    print(csv_path)
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    for line in lines:
        source_path = line[0]
        filename = source_path.split('/')[-1]
        img_path = current_path + 'IMG/' + filename
        image = cv2.imread(img_path)
#        image = cv2.cvtColor( image, cv2.COLOR_RGB2GRAY)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)


for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement * -1.0)
    
X_train = np.array(augmented_images)
#X_train = X_train.reshape(X_train.shape[0], 160, 320, 1)
y_train = np.array(augmented_measurements)    

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Lambda,Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = X_train.shape[1:]))
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape = X_train.shape[1:]))
model.add(Convolution2D(24, 5, 5,subsample = (2,2), activation = 'relu'))
#model.add(Dropout(0.5))
model.add(Convolution2D(36, 5, 5,subsample = (2,2), activation = 'relu'))
#model.add(Dropout(0.5))
model.add(Convolution2D(48, 5, 5,subsample = (2,2), activation = 'relu'))
#model.add(Dropout(0.5))
model.add(Convolution2D(64,3,3,activation = 'relu'))
model.add(Convolution2D(64,3,3,activation = 'relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True, nb_epoch=5)

model.save('model.h5')
