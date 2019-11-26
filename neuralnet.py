
# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import glob
np.random.seed(123)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle as pkl
import time
# Load data into train and test sets

#THIS NEEDS TO BE HANDLED DIFFERENTLY
num_authors = 20

# # map data to input and output arrays
# x = []
# y = []
# for direct in glob.iglob('data/*/'):
#     author = direct.split('\\')[1]
#     for filepath in glob.iglob(direct + '*.png'):
#         x.append(mpimg.imread(filepath))
#         y.append(author)
# x = np.asarray(x)
# y = np.asarray(y)
# pkl.dump(x, open("inputs.pkl",'wb'))
# pkl.dump(y, open('outputs.pkl', 'wb'))

# x = pkl.load(open("inputs.pkl", 'rb'))
# y = pkl.load(open("outputs.pkl", 'rb'))
#
# encoder = LabelEncoder()
# encoder.fit(y)
# encoded_y = encoder.transform(y)
#
# # CHECK IF THIS IS RANDOM, may need to pickle set, or comment out and load the model
# X_train, X_test, y_train, y_test = train_test_split(x, encoded_y, test_size=0.167, random_state=1)
#
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
#
# # maybe pickle data?
# pkl.dump(X_train, open("X_train.pkl", 'wb'))
# pkl.dump(y_train, open("y_train.pkl", 'wb'))
# pkl.dump(X_test, open("X_test.pkl", 'wb'))
# pkl.dump(y_test, open("y_test.pkl", 'wb'))
# pkl.dump(X_val, open("X_val.pkl", 'wb'))
# pkl.dump(y_val, open("y_val.pkl", 'wb'))

X_train = pkl.load(open("X_train.pkl", "rb"))
y_train = pkl.load(open("y_train.pkl", "rb"))
X_test = pkl.load(open("X_test.pkl", "rb"))
y_test = pkl.load(open("y_test.pkl", "rb"))
X_val = pkl.load(open("X_val.pkl", "rb"))
y_val = pkl.load(open("y_val.pkl", "rb"))

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(X_val.shape, y_val.shape)


# maybe only for Theano, changes depth to 1
X_train = X_train.reshape(X_train.shape[0], 1, 113, 113)
X_test = X_test.reshape(X_test.shape[0], 1, 113, 113)
print(X_train.shape)
#
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# NOTE: ONLY 20 b/c that is how many authors we are testing
# 20 SHOULD BE A VARIABLE

y_train = np_utils.to_categorical(y_train, num_authors)
y_test = np_utils.to_categorical(y_test, num_authors)
print(y_train.shape)

# Model Architecture: This is where most of the work is
model = Sequential()
#relu is Rectified Linear Unit
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1,113,113), data_format='channels_first'))
print(model.output_shape)

#more layers
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25)) #regularizing to prevent overfitting

# add fully connected layer and output layer
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_authors, activation='softmax'))
print(model.output_shape)
# compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=32, nb_epoch=10, verbose=1)
# save model
filename = 'finalized_model.sav'
pkl.dump(model, open(filename, 'wb'))

# evaluate model
score = model.evaluate(X_test, y_test, verbose=0)
print(score)
