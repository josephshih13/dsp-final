import numpy as np
np.set_printoptions(precision = 6, suppress = True)
import csv
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils, plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Activation, LeakyReLU
import time
from keras.models import load_model

X = np.load('X.npy')
Y = np.load('Y.npy')
test_x = X[:200]
test_x = test_x/255
test_x = test_x.reshape(200,64,64,1)
test_y = Y[:200]

model = load_model(sys.argv[1])

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
score, acc = model.evaluate(test_x,test_y)
print(acc)



