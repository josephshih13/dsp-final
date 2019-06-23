import cv2
import numpy as np
import sys
from os import listdir
from os.path import isfile, join
from keras.utils import np_utils

file_names = [f for f in listdir('./spectrograms') if '.png' in f]
print(file_names[0])

X = []
tmp = []

for f in file_names:
    
    img = cv2.imread('./spectrograms/'+f,0)
    #np.set_printoptions(threshold=sys.maxsize)
    #print(img)

    trunc_img = img[4:-3,4:-3]
    #print(trunc_img)
    X.append(trunc_img)
    tmp.append(int(f[0]))

Y = np_utils.to_categorical(tmp, 10)

print(Y[0])

X = np.array(X)

indices = np.arange(X.shape[0])
np.random.shuffle(indices)

X = X[indices]
Y = Y[indices]

np.save('X.npy',X)
np.save('Y.npy',Y)


