from keras.layers import Activation, Convolution2D, Dropout
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Input
from keras import layers
from keras.regularizers import l2
#For AVX2 Error
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
def nothing(x):
    pass

def simple_CNN(input_shape, num_classes):
    classifier = Sequential()
    # Step 1.1 Convulution Layer
    classifier.add(Convolution2D(filters=16, kernel_size=(7, 7), padding='same',
                                name='image_array', input_shape=input_shape))
    classifier.add(BatchNormalization())
    classifier.add(Convolution2D(filters=16, kernel_size=(7, 7), padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    # Step 1.2 Pooling2D
    classifier.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    classifier.add(Dropout(.5))
    # Step 2.1 Convulution Layer
    classifier.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    # Step 2.2 Pooling2D
    classifier.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    classifier.add(Dropout(.5))

    classifier.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
    # Step Relu Layer
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    # Average pooling layer
    classifier.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    #overfitting
    classifier.add(Dropout(.5))
    # 3.1 Convulution Layer
    classifier.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
    #relu layer
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    # 3.2 Pooling2D
    classifier.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    classifier.add(Dropout(.5))

    classifier.add(Convolution2D(filters=256, kernel_size=(3, 3), padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Convolution2D(filters=num_classes, kernel_size=(3, 3), padding='same'))
    # GlobalAveragePooling2D does something different.
    # It applies average pooling on the spatial dimensions until each spatial dimension is one,
    # and leaves other dimensions unchanged.
    # In this case values are not kept as they are averaged.
    classifier.add(GlobalAveragePooling2D())
    classifier.add(Activation('softmax', name='predictions'))
    return classifier

def simpler_CNN(input_shape, num_classes):

    classifier = Sequential()
    classifier.add(Convolution2D(filters=16, kernel_size=(5, 5), padding='same',name='image_array', input_shape=input_shape))

    classifier.add(BatchNormalization())
    classifier.add(Convolution2D(filters=16, kernel_size=(5, 5),strides=(2, 2), padding='same'))

    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(Dropout(.25))

    classifier.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Convolution2D(filters=32, kernel_size=(5, 5), strides=(2, 2), padding='same'))

    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(Dropout(.25))

    classifier.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Convolution2D(filters=64, kernel_size=(3, 3),
                                 strides=(2, 2), padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(Dropout(.25))

    classifier.add(Convolution2D(filters=64, kernel_size=(1, 1), padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Convolution2D(filters=128, kernel_size=(3, 3),
                                 strides=(2, 2), padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(Dropout(.25))

    classifier.add(Convolution2D(filters=256, kernel_size=(1, 1), padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Convolution2D(filters=128, kernel_size=(3, 3),
                                 strides=(2, 2), padding='same'))

    classifier.add(Convolution2D(filters=256, kernel_size=(1, 1), padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Convolution2D(filters=num_classes, kernel_size=(3, 3),
                                 strides=(2, 2), padding='same'))
    # Flatten
    classifier.add(Flatten())
    classifier.add(Activation('softmax', name='predictions'))
    return classifier