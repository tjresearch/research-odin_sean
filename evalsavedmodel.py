# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import glob
np.random.seed(123)

# from tensorflow.keras.optimizers import SGD, Adam
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
# from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
# from tensorflow.python.keras.utils import np_utils
# from tensorflow.python.keras import backend as K
# from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation, BatchNormalization
from tensorflow.python.keras.layers.convolutional import Convolution2D, Cropping2D, ZeroPadding2D, MaxPooling2D
import tensorflow as tf
# from keras.optimizers import SGD, Adam, RMSprop
#
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import pickle as pkl
import time

#Setting up data
num_authors = 20

X_test = pkl.load(open("X_test.pkl", "rb"))
y_test = pkl.load(open("y_test.pkl", "rb"))
X_test = X_test.reshape(X_test.shape[0], 1, 113, 113)
X_test = X_test.astype('float32')
X_test /= 255
y_test = to_categorical(y_test, num_authors)

# Recreate the exact same model purely from the file
model = tf.keras.models.load_model('modelsave')
score = model.evaluate(X_test, y_test, verbose=0)
print(score)

np.set_printoptions(linewidth = 200)
Y_pred = model.predict(X_test)
print(Y_pred.shape)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred.shape)
y_acc = np.argmax(y_test, axis=1)
print(y_pred)
print(y_acc)

# tf.compat.v1.disable_eager_execution()
# cm = tf.math.confusion_matrix(y_acc, y_pred)
# sess = tf.compat.v1.Session()
# with sess.as_default():
#     print(cm.eval())