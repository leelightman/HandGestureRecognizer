## CNN model

import numpy as np
import os
from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import json
import cv2
import matplotlib
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from keras import backend as K

dic ={"f":0, "h":1, "o":2, "p":3, "v":4}
path_to_training_data = ["HandGestureRecognizer/data/f",
                        "HandGestureRecognizer/data/h",
                        "HandGestureRecognizer/data/o",
                         "HandGestureRecognizer/data/p",
                        "HandGestureRecognizer/data/v"]

NUM_OF_GESTURES = 5
NUM_OF_LAYERS = 32
KERNEL_SIZE = 3
SIZE =2000
img_rows, img_cols = 286, 300
# img_rows, img_cols = 300, 32
totalImg = []
s = 0
label=np.ones((SIZE,),dtype = int)
im =  np.array(Image.open("HandGestureRecognizer/data/f/f2_1.jpg"))

m,n = im.shape[0:2]
print(m,n)

for path in path_to_training_data:
    listing = os.listdir(path)
    retlist = []
    for name in listing:
        retlist.append(name)
    immatrix = np.array([np.array(Image.open(path+ '/' + images).convert('L')).flatten()
                     for images in sorted(retlist)], dtype = 'f')
    totalImg.extend(immatrix)
    label[s:s+len(immatrix)] = dic[path[-1]]
    s += len(immatrix)

totalImg = np.array(totalImg)


data,Label = shuffle(totalImg,label, random_state=2)
train_data = [data,Label]

(X, y) = (train_data[0],train_data[1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
X_train = X_train.reshape(X_train.shape[0], 1, m, n)
X_test = X_test.reshape(X_test.shape[0], 1, m, n)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, NUM_OF_GESTURES)
Y_test = np_utils.to_categorical(y_test, NUM_OF_GESTURES)

model = Sequential()

model.add(Conv2D(NUM_OF_LAYERS, (KERNEL_SIZE, KERNEL_SIZE),
                    padding='valid',
                    input_shape=(m,n,1)))
convout1 = Activation('relu')
model.add(convout1)
model.add(Conv2D(NUM_OF_LAYERS, (KERNEL_SIZE, KERNEL_SIZE)))
convout2 = Activation('relu')
model.add(convout2)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_OF_GESTURES))
model.add(Activation('softmax'))


