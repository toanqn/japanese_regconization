# This code is based on
# https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
# https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py

import numpy as np
import scipy.misc
from keras import backend as K
from keras import initializers
# from keras import initializations
from keras.layers import Dense, Conv2D, Input, MaxPooling2D, Flatten, Dropout, Activation
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

nb_classes = 3036
# input image dimensions
img_rows, img_cols = 32, 32
# img_rows, img_cols = 127, 128

# ary = np.load("data_kanji.npz")['arr_0'].reshape([-1, 64, 64]).astype(np.float32) / 15
# ary = np.load("data.npz")['arr_0'].reshape([-1, 64, 64]).astype(np.float32) / 15
input_shape = np.load("data.npz")['input_shape']
X_train = np.load("data.npz")['X_train']
Y_train = np.load("data.npz")['Y_train']
X_test = np.load("data.npz")['X_test']
Y_test = np.load("data.npz")['Y_test']

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

datagen = ImageDataGenerator(rotation_range=15, zoom_range=0.20)
datagen.fit(X_train)

model = Sequential()


# def my_init():
#     # return initializations.normal(shape, scale=0.1, name=name)
#     return initializers.RandomNormal(stddev=0.1)


# Best val_loss: 0.0205 - val_acc: 0.9978 (just tried only once)
# 30 minutes on Amazon EC2 g2.2xlarge (NVIDIA GRID K520)
def m6_1():
    model.add(Conv2D(32, 3, init=initializers.RandomNormal(stddev=0.1), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, 3, init=initializers.RandomNormal(stddev=0.1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, 3, init=initializers.RandomNormal(stddev=0.1)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, 3, init=initializers.RandomNormal(stddev=0.1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(4096, init=initializers.RandomNormal(stddev=0.1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))


def classic_neural():
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))


m6_1()
# classic_neural()

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=128), samples_per_epoch=X_train.shape[0],
                    nb_epoch=50, validation_data=(X_test, Y_test))
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
	json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
