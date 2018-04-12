# Rohan Borkar
# Convolutional neural network
# For big datasets. Like 10k+ sample stuff

# Import keras
import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras import backend as K
# Preprocessing. AKA loading images to use
from keras.preprocessing.image import ImageDataGenerator
# Layer and optimization
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
# Cifar is the dataset
from keras.datasets import cifar10
# Numpy is bae <3
import numpy as np

#=================================================================================================================================D
# The following are parameters for the import data and model. I like to keep them at the top

num_classes = 2                                                             # Number of classes should the model should distinguish
baseMapNum = 32                                                             # No idea what this is loool. Used as 'filter' in Conv2D
img_width, img_height = 1024, 1024                                          # Width and height of images in dataset
weight_decay = [1e-4]                                                       # Try different values from 0.0001 to 10 (I think?)
img_shape = (1024, 1024, 3)                                                   # Shape of the image (width, height, depth)
epochs = 3                                                                  # How many iterations to do. This is multiplied by 5
batch_size = 2                                                              # Increase or decrease depending on memory. Higher the better

# Data directories and details
train_data_dir = 'data/train'                                               # Location of training data (for training the model)
#validation_data_dir = 'data/validation'                                     # Location of validation data (for estimating training quality)
test_data_dir = 'data/test'                                                 # Location of testing data (for application)
nb_train_samples = 10                                                       # How many samples to use in testing
nb_validation_samples = 4                                                   # How many samples to use in testing

#=================================================================================================================================D
# Here's our super basic model. It's in a fx in case you want to use multiple runs for optimization and stuff. Will add more than weight decay
# It doesn't look basic, but it is. Yeah you weren't expecting that, kiddo. Even I don't understand this shit, how do you expect to? Get shrekt m9

def make_run_model(weight_decay):
    model = Sequential()
    model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=img_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()


# Running the model. Man, I really don't understand any of this lol
# Designed to run from a folder of images

    # Augmentation configuration we will use for training.
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False
        )

    # Augmentation configuration we will use for testing. Basically making sure it doesn't mess with the image at all
    test_datagen = ImageDataGenerator(
        rescale=1. / 255
        )

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    # Training

    # Run 1
    # With optimization, compilation, and generation
    opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
    model.compile(loss='categorical_crossentropy',
            optimizer=opt_rms,
            metrics=['accuracy'])
    ##############################
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs*3,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)
    #model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),steps_per_epoch=x_train.shape[0] // batch_size,epochs=3*epochs,verbose=1,validation_data=(x_test,y_test))
    model.save_weights('NHIcxr_ep75_' + str(weight_decay) + '.h5')


    # Run 2
    # With optimization, compilation, and generation
    opt_rms = keras.optimizers.rmsprop(lr=0.0005,decay=1e-6)
    model.compile(loss='categorical_crossentropy',
            optimizer=opt_rms,
            metrics=['accuracy'])
    #############################
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)
    #model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),steps_per_epoch=x_train.shape[0] // batch_size,epochs=epochs,verbose=1,validation_data=(x_test,y_test))
    model.save_weights('NHIcxr_ep100_' + str(weight_decay) + '.h5')

    # Run 3
    # With optimization, compilation, and generation
    opt_rms = keras.optimizers.rmsprop(lr=0.0003,decay=1e-6)
    model.compile(loss='categorical_crossentropy',
            optimizer=opt_rms,
            metrics=['accuracy'])
    ############################
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)
    #model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),steps_per_epoch=x_train.shape[0] // batch_size,epochs=epochs,verbose=1,validation_data=(x_test,y_test))
    model.save_weights('NHIcxr_ep125_' + str(weight_decay) + '.h5')

    #testing - no kaggle eval
    scores = model.evaluate(test_datagen, batch_size=128, verbose=1)
    print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))


for val in weight_decay:
    make_run_model(val)
