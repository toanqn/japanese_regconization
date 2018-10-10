import numpy as np
import h5py, time
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib

from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv2D, Input, MaxPooling2D, Flatten, Dropout, Activation
from keras.optimizers import Adadelta

matplotlib.use('Agg')

#Training generator with augmentation

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

# Testing generator

test_datagen = ImageDataGenerator(rescale=1./255)

BATCH_SIZE = 16
N_CLASSES = 3036
LR = 0.0001
N_EPOCHS = 40
IMG_SIZE = 64

train_generator = train_datagen.flow_from_directory(
	'training',  # this is the target directory
	target_size=(IMG_SIZE, IMG_SIZE),  # all images will be resized to 150x150
	batch_size=BATCH_SIZE,
	class_mode='sparse')

validation_generator = test_datagen.flow_from_directory(
	'testing',
	target_size=(IMG_SIZE, IMG_SIZE),
	batch_size=BATCH_SIZE,
	class_mode='sparse')

# from keras.applications.mobilenet import MobileNet
# model = MobileNet(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=True, classes=N_CLASSES, weights=None)

from keras.layers.normalization import BatchNormalization

model=Sequential()
# #layer Conv2D_1
model.add(Conv2D(64, 3, padding="same", input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Conv2D(128, 3, padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Dropout(0.5))
# #layer Conv2D_2
model.add(Conv2D(512, 3, padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, 3, padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Dropout(0.5))
# model.add(Conv2D(512,3,padding="same",activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Flatten())
# #layer Dense 1
model.add(Dense(units=4096))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
# #layer Dense 2
model.add(Dense(N_CLASSES))
model.add(BatchNormalization())
model.add(Activation('softmax'))

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
	train_generator,
	steps_per_epoch=60720 // BATCH_SIZE,   #576840 // BATCH_SIZE,
	epochs=N_EPOCHS,
	validation_data=validation_generator,
	validation_steps=6072 // BATCH_SIZE,   #30360 // BATCH_SIZE,
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