import numpy as np
import h5py, time
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib
import pickle
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv2D, Input, MaxPooling2D, Flatten, Dropout, Activation
from keras.utils import np_utils
from keras import backend as K

from sklearn.model_selection import train_test_split

BATCH_SIZE = 256
N_CLASSES = 3036
LR = 0.001
N_EPOCHS = 40
IMG_SIZE = 64

matplotlib.use('Agg')

#Training generator with augmentation
# with open('data.pickle', 'rb') as f:
# 	X_train, Y_train = pickle.load(f)

data = np.load('data.npz')
X_train = data['images']
Y_train = data['labels']
data.close()

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2)

# if K.image_dim_ordering() == 'th':
#     X_train = X_train.reshape(X_train.shape[0], 1, IMG_SIZE, IMG_SIZE)
#     X_test = X_test.reshape(X_test.shape[0], 1, IMG_SIZE, IMG_SIZE)
#     input_shape = (1, IMG_SIZE, IMG_SIZE)
# else:
#     X_train = X_train.reshape(X_train.shape[0], IMG_SIZE, IMG_SIZE, 1)
#     X_test = X_test.reshape(X_test.shape[0], IMG_SIZE, IMG_SIZE, 1)
#     input_shape = (IMG_SIZE, IMG_SIZE, 1)

Y_train = np_utils.to_categorical(Y_train, N_CLASSES)
y_test = np_utils.to_categorical(Y_test, N_CLASSES)

train_datagen = ImageDataGenerator(
	rotation_range=40,
	width_shift_range=0.2,
	height_shift_range=0.2,
	rescale=1./255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=False,
	vertical_flip=False,
	fill_mode='nearest')

train_datagen.fit(X_train)

# from keras.applications.mobilenet import MobileNet
# model = MobileNet(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=True, classes=N_CLASSES, weights=None)

model=Sequential()
# #layer Conv2D_1
model.add(Conv2D(64, 3, padding="same", activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Conv2D(128, 3, padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Dropout(0.5))
# #layer Conv2D_2
model.add(Conv2D(512, 3, padding="same", activation='relu'))
model.add(Conv2D(512, 3, padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Dropout(0.5))
# model.add(Conv2D(512,3,padding="same",activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Flatten())
# #layer Dense 1
model.add(Dense(units=4096,activation='relu'))
model.add(Dropout(0.5))
# #layer Dense 2
model.add(Dense(N_CLASSES, activation='softmax'))

model.summary()

# Training

from keras.callbacks import ModelCheckpoint
model_file = "models_4/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"

checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.compile(loss='sparse_categorical_crossentropy',
	optimizer='adadelta',
	metrics=['accuracy'])

start_time=time.time()
history = model.fit_generator(
	train_datagen.flow(X_train, Y_train),
	samples_per_epoch=X_train.shape[0],
	epochs=N_EPOCHS,
	validation_data=(X_test, Y_test),
	# validation_steps=6072 // BATCH_SIZE,   #30360 // BATCH_SIZE,
	callbacks=callbacks_list)

#evaluation
end_time= time.time()
total_time = (end_time - start_time)
print("Time to training: ", total_time, " seconds")

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
	json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

print(history.history.keys)
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'], loc='upper left')
plt.savefig("model_accuracy")
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss'], loc='upper left')
plt.savefig("model_loss")
plt.show()