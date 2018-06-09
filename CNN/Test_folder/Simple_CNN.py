# Sky Trish ???

# Rohan Borkar
# Neural network
# For big image datasets. Like 10k+ samples.

# Import keras
import keras
from keras.models import Sequential
from keras import backend as K
# Preprocessing. AKA loading images to use
from keras.preprocessing.image import ImageDataGenerator
# Layer and optimization
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
# Visualisation
import matplotlib.pyplot as plt
# Numpy is bae <3
import numpy as np
import random
import os.path
import pandas as pd

#=================================================================================================================================D
# The following are parameters for the import data and model. I like to keep them at the top

model_name = 'NHIcxr'
num_classes = 2                                                             # Number of classes should the model should distinguish
img_width, img_height = 1024, 1024                                          # Width and height of images in dataset
img_shape = (1024, 1024, 3)                                                 # Shape of the image (width, height, depth)
max_queue_num = 2                                                           # Maximum size of generator queue
nb_train_samples = 8                                                        # How many samples to use in training per step
nb_validation_samples = 4                                                   # How many samples to use in validation per step


# Data directories and details
train_data_dir = 'data/train'                                               # Location of training data (for training the model)
validation_data_dir = 'data/validation'                                     # Location of validation data (for estimating training quality)
test_data_dir = 'data/test'                                                 # Location of testing data (for application)

param_names = ['activation1', 'activation2', 'activation3', 'activation4', 'activation5', 'activation6',
               'decay',       'optimizer',   'batch_size',  'dropout1',    'dropout2',    'dropout3',
               'epochs',      'lr1',         'baseMapNum',  'run',         'accuracy',    'loss'        ]
#=================================================================================================================================D
# Here are functions that produce/run models and parameters. Want me to explain this shit? Too bad, I can't. Get shrekt, m9

# Give model_generator will make a model with specific hyperparameters
def model_generator( parameters ):

    # Unpackage parameters
    activation1, activation2, activation3, activation4, activation5, activation6, decay, _, _, dropout1, dropout2, dropout3, _, _, baseMapNum = parameters

    model = Sequential()
    model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(decay), input_shape=img_shape))
    model.add(Activation(activation1))
    model.add(BatchNormalization())
    model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(decay)))
    model.add(Activation(activation2))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(dropout1))

    model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(decay)))
    model.add(Activation(activation3))
    model.add(BatchNormalization())
    model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(decay)))
    model.add(Activation(activation4))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(dropout2))

    model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(decay)))
    model.add(Activation(activation5))
    model.add(BatchNormalization())
    model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(decay)))
    model.add(Activation(activation6))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(dropout3))

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    return model

#=================================================================================================================================D
# Running the model. Man, I don't understand any of this lol
# Designed to run from a folder of images
    # Augmentation configuration we will use for training.
def run_model(model, parameters):
    # Unpackage parameters
    _, _, _, _, _, _, _, optimizer, batch_size, _, _, _, epochs, lr1, _ = parameters

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

    # Augmentation configuration we will use for testing. Basically don't mess with the image
    test_datagen = ImageDataGenerator(
        rescale=1. / 255
        )

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
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
    model_optimizer = str('model_optimizer = keras.optimizers.' + optimizer + '(lr=' + str(lr1) + ', decay=1e-6)')
    exec(model_optimizer)
    model.compile(loss='categorical_crossentropy',
            optimizer=model_optimizer,
            metrics=['accuracy'])
    hist1 = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs*3,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

    # Run 2
    # With optimization, compilation, and generation
    model_optimizer = str('model_optimizer = keras.optimizers.' + optimizer + '(lr=' + str(lr1/2) + ', decay=1e-6)')
    exec(model_optimizer)
    model.compile(loss='categorical_crossentropy',
            optimizer=model_optimizer,
            metrics=['accuracy'])
    hist2 = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

    # Run 3
    # With optimization, compilation, and generation
    model_optimizer = str('model_optimizer = keras.optimizers.' + optimizer + '(lr=' + str(lr1/3) + ', decay=1e-6)')
    exec(model_optimizer)
    model.compile(loss='categorical_crossentropy',
            optimizer=model_optimizer,
            metrics=['accuracy'])
    hist3 = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)
    model.save_weights(model_name + '_ep' + str(epochs*5) + '.h5')

    # Testin with test folder data
    score = model.evaluate_generator(
        test_generator, 
        steps=10)
    print (score)
    # Returning only hist1 for now, will write code to join 1-3 later
    return score, hist1


def run():
    # Generate parameters. Uses randomization
    print ("\nCreating parameters...")
    params = ['relu', 'relu', 'relu', 'relu', 'relu', 'relu',    # Activations
              1e-6, 'adam', 2, 0.3, 0.4, 0.5, 1, 0.001, 2 ]   # Decay, optimizer, batch size, dropouts, epochs, learning rate, baseMapNum
    # Make model based on parameters
    print ("\nCreating model...")
    model = model_generator(params)
    # Run model with parameters
    print ("\nRunning model...")
    score, history = run_model(model, params)
    print ("\nRecording results...")
    # Add parameters + results to past_params
    # params.append(run)
    # params.append(score[1])
    # params.append(score[0])
    print score
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # global param_names
    # global past_params
    # if not os.path.isfile(model_name + '_results.csv'):
    #     past_params = pd.DataFrame(np.array(params).reshape(1,18), columns = param_names)
    #     past_params.to_csv(model_name + "_results.csv")
    # elif os.path.isfile(model_name + '_results.csv'):
    #     params = np.array(params)
    #     past_params = past_params.values
        
    #     past_params = pd.DataFrame(past_params, columns = param_names)
    #     past_params.to_csv(model_name + '_results.csv')
    #print ("Run completed with " + str(params[16]*100) + " percent accuracy")


run()


"""
To do:
Add dataframe + csv support
Add optimizer to run_model
"""