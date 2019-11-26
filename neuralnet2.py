
# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import glob
np.random.seed(123)

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Add
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Input
from keras.layers import AveragePooling2D
from keras.utils import np_utils
from keras.initializers import glorot_uniform
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

def identity_block(X, f, filters, stage, block):
    #Name base
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    #extract filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    #Three components:
    X = Convolution2D(filters = F1, kernel_size = (1,1),strides=(1,1),padding='valid',
               name=conv_name_base+'2a',kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name = bn_name_base+'2a')(X)
    X = Activation('relu')(X)

    X = Convolution2D(filters = F2, kernel_size = (f,f),strides=(1,1),padding='same',
               name=conv_name_base+'2b',kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name = bn_name_base+'2b')(X)
    X = Activation('relu')(X)

    X = Convolution2D(filters = F3, kernel_size = (1,1),strides=(1,1),padding='valid',
               name=conv_name_base+'2c',kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name = bn_name_base+'2c')(X)

    #Add shortcut
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):

    #Name base
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    #extract filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ###MAIN PATH###
    X = Convolution2D(filters = F1, kernel_size = (1,1), strides = (s,s), padding= 'valid',
               name = conv_name_base+'2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name = bn_name_base+'2a')(X)
    X = Activation('relu')(X)

    X = Convolution2D(filters = F2, kernel_size = (f,f),strides=(1,1),padding='same',
               name=conv_name_base+'2b',kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name = bn_name_base+'2b')(X)
    X = Activation('relu')(X)

    X = Convolution2D(filters = F3, kernel_size = (1,1),strides=(1,1),padding='valid',
               name=conv_name_base+'2c',kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name = bn_name_base+'2c')(X)

    ###SHORTCUT PATH###
    X_shortcut = Convolution2D(filters=F3, kernel_size=(1,1), strides=(s,s),
                               padding = 'valid', name = conv_name_base + '1',
                               kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name= bn_name_base+'1')(X_shortcut)

    #Add shortcut
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    return X


def ResNet50(input_shape=(113, 113,1), classes=num_authors):
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Convolution2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL
    X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)

    # Output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model







# maybe only for Theano, changes depth to 1
X_train = X_train.reshape(X_train.shape[0], 113, 113, 1)
X_test = X_test.reshape(X_test.shape[0], 113, 113, 1)
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

# Model Architecture: This is where most of the work is
model = ResNet50(input_shape=(113,113,1), classes=num_authors)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=32, epochs=2)
# save model
filename = 'finalized_model.sav'
pkl.dump(model, open(filename, 'wb'))

# evaluate model
score = model.evaluate(X_test, y_test, verbose=0)
print(score)
