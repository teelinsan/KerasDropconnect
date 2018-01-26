'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import tensorflow as tf
tf.set_random_seed(1337)
from keras.models import Model
from keras.layers import Input, Dense, Dropout

from ddrop.layers import DropConnectDense, DropConnect
from utils import mnist

batch_size = 128
nb_epoch = 12

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

(X_train, Y_train), (X_test, Y_test), nb_classes = mnist()

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
inputs = Input(shape=(np.prod(X_train.shape[1:]),))
x = Dense(128, activation='relu')(inputs)
# x = Dense(64, activation='relu')(x)
x = DropConnect(Dense(64, activation='relu'), prob=0.5)(x)
# x = Dropout(0.6)(x)
predictions = Dense(nb_classes, activation='softmax')(x)

model = Model(input=inputs, output=predictions)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
