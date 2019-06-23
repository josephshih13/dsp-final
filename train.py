import numpy as np
np.set_printoptions(precision = 6, suppress = True)
import csv
# from sys import argv
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

SHAPE = 48
CATEGORY = 7

MODEL_DIR = './model'
HIS_DIR = './history'

READ_FROM_NPZ = 0
AUGMENT = 1

def read_train(filename):

    X, Y = [], []
    with open(filename, 'r', encoding='big5') as f:
        count = 0
        for line in list(csv.reader(f))[1:]:
            Y.append( float(line[0]) )
            X.append( [float(x) for x in line[1].split()] )
            count += 1
            print('\rX_train: ' + repr(count), end='', flush=True)
        print('', flush=True)

    return np.array(X), np_utils.to_categorical(Y, CATEGORY)

# argv: [1]train.csv
def main():

    print('============================================================')

    X = np.load('X.npy')
    Y = np.load('Y.npy')


    print('Reshape data')
    X = X/255
    X = X.reshape(2000,64,64,1)
    X = X[:-200]
    Y = Y[:-200]
    print(X.shape)


    print('============================================================')
    print('Construct model')
    model = Sequential()
    model.add(Conv2D(128, input_shape=(64, 64, 1), kernel_size=(3, 3), activation='relu', padding='same',
                     kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))

    model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.35))

    model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.35))

    model.add(Flatten())

    model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax', kernel_initializer='glorot_normal'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())
    print('============================================================')
    VAL = 100
    BATCH = 128
    EPOCHS = 200

    print('Train with raw data')
    cp = ModelCheckpoint('temp_model.h5', monitor='val_acc', verbose=0, save_best_only=True)
    history = model.fit(X, Y, batch_size=BATCH, epochs=EPOCHS, verbose=1, validation_split=0.1, callbacks=[cp])


    print('============================================================')
    print('Evaluate train')
    score = model.evaluate(X, Y)
    score = '{:.6f}'.format(score[1])
    print('Train accuracy (all):', score)

    print('============================================================')
    # print('Save model')
    # model.save(MODEL_DIR + '/' + score + '.h5')
    H = history.history
    best_val = '{:.6f}'.format(np.max(H['val_acc']))
    last_val = '{:.6f}'.format(H['val_acc'][-1])
    print('Best val: ' + best_val)
    print('Last val: ' + last_val)
    print('Save best model')
    os.rename('temp_model.h5', MODEL_DIR + '/' + best_val + '.h5')
    print('Save last model')
    model.save(MODEL_DIR + '/' + last_val + '.h5')


if __name__ == '__main__':
    main()