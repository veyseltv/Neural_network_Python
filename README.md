# Convolution_Neural_Network_Python

Convolution Neural network sample layer structure with python.

# KERAS

  Keras is an open-source neural network library written in Python. It is capable of running on top of TensorFlow, Microsoft Cognitive Toolkit, R, Theano, or PlaidML. Designed to enable fast experimentation with deep neural networks, it focuses on being user-friendly, modular, and extensible.
  
# Convolution Neural Network (CNN)

  Convolutional layers are the major building blocks used in convolutional neural networks.

A convolution is the simple application of a filter to an input that results in an activation. Repeated application of the same filter to an input results in a map of activations called a feature map, indicating the locations and strength of a detected feature in an input, such as an image.

The innovation of convolutional neural networks is the ability to automatically learn a large number of filters in parallel specific to a training dataset under the constraints of a specific predictive modeling problem, such as image classification. The result is highly specific features that can be detected anywhere on input images.


![CNN Visualization jpg](https://user-images.githubusercontent.com/59453566/93585013-d4522380-f9ae-11ea-818e-bfbb2e60b3b3.jpeg)


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
