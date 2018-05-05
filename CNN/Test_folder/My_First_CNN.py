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
# Numpy is bae <3
import numpy as np
import random

#=================================================================================================================================D
# The following are parameters for the import data and model. I like to keep them at the top

model_name = 'NHIcxr'
num_classes = 15                                                            # Number of classes should the model should distinguish
img_width, img_height = 1024, 1024                                          # Width and height of images in dataset
img_shape = (1024, 1024, 3)                                                 # Shape of the image (width, height, depth)
max_queue_num = 2                                                           # Maximum size of generator queue
nb_train_samples = 8                                                        # How many samples to use in training per step
nb_validation_samples = 4                                                   # How many samples to use in validation per step


# Data directories and details
train_data_dir = 'data/train'                                               # Location of training data (for training the model)
validation_data_dir = 'data/validation'                                     # Location of validation data (for estimating training quality)
test_data_dir = 'data/test'                                                 # Location of testing data (for application)

# Hyperparameters
activations = ['relu', 'tanh', 'sigmoid']                           # What kind of curve to use for activation layers
baseMapNums = [32,64,128,256,512,1024]                                      # No idea what this is loool. Used as 'filter' in Conv2D
weight_decays = [1e-6, 1e-4, 1e-2, 1, 10]
dropouts = [0.2, 0.3, 0.4, 0.5]                                             # I dunno man, Rah says it makes the model "turn off its brain"
optimizers = ['rmsprop', 'adam', 'adagrad', 'sgd']                          # Optimizers to optimize...stuff. I really have no idea
#epochs = [20, 25, 40]                                                       # How many iterations to do. This is multiplied by 5
epochs = [1, 2]                                                             # Low number of epochs for testing purposes only
#batch_size = [32, 64, 128]                                                  # Increase or decrease depending on memory. Higher the better                                                              
batch_sizes = [2, 4]                                                        # Small batches for testing purposes only
param_grid = dict( activation1 = activations, activation2 = activations, activation3 = activations, 
                   activation4 = activations, activation5 = activations, activation6 = activations, 
                   decay1 = weight_decays,    decay2 = weight_decays,    decay3 = weight_decays,
                   decay4 = weight_decays,    decay5 = weight_decays,    decay6 = weight_decays,
                   dropout1 = dropouts,       dropout2 = dropouts,       dropout3 = dropouts,
                   epochs = epochs,           batch_size = batch_sizes,  baseMapNum = baseMapNums   )
past_params = []                                                            # Saves all sets of parameters of trials
past_scores = []                                                            # Saves all score reorts of trials
current_params = dict()
runs = 0

#=================================================================================================================================D
# Here are functions that produce/run models and parameters. Want me to explain this shit? Too bad, I can't. Get shrekt, m9

# Give model_generator will make a model with specific hyperparameters
def model_generator( param_dict ):        # Epochs is 1 for test purposes
    model = Sequential()
    model.add(Conv2D(param_dict['baseMapNum'], (3,3), padding='same', kernel_regularizer=regularizers.l2(param_dict['decay1']), input_shape=img_shape))
    model.add(Activation(param_dict['activation1']))
    model.add(BatchNormalization())
    model.add(Conv2D(param_dict['baseMapNum'], (3,3), padding='same', kernel_regularizer=regularizers.l2(param_dict['decay2'])))
    model.add(Activation(param_dict['activation2']))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(param_dict['dropout1']))

    model.add(Conv2D(2*param_dict['baseMapNum'], (3,3), padding='same', kernel_regularizer=regularizers.l2(param_dict['decay3'])))
    model.add(Activation(param_dict['activation3']))
    model.add(BatchNormalization())
    model.add(Conv2D(2*param_dict['baseMapNum'], (3,3), padding='same', kernel_regularizer=regularizers.l2(param_dict['decay4'])))
    model.add(Activation(param_dict['activation4']))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(param_dict['dropout2']))

    model.add(Conv2D(4*param_dict['baseMapNum'], (3,3), padding='same', kernel_regularizer=regularizers.l2(param_dict['decay5'])))
    model.add(Activation(param_dict['activation5']))
    model.add(BatchNormalization())
    model.add(Conv2D(4*param_dict['baseMapNum'], (3,3), padding='same', kernel_regularizer=regularizers.l2(param_dict['decay6'])))
    model.add(Activation(param_dict['activation6']))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(param_dict['dropout3']))

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()
    return model

#=================================================================================================================================D
# Running the model. Man, I don't understand any of this lol
# Designed to run from a folder of images
    # Augmentation configuration we will use for training.
def run_model(model, batch_size=4):
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



    print nb_validation_samples
    print batch_size
    # Training
    # Run 1
    # With optimization, compilation, and generation
    opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
    model.compile(loss='categorical_crossentropy',
            optimizer=opt_rms,
            metrics=['accuracy'])
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs*3,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)
    model.save_weights(model_name + '_ep' + epochs*3 + '.h5')
    scores1 = model.evaluate_generator(
        test_generator, 
        steps=10)

    # Run 2
    # With optimization, compilation, and generation
    opt_rms = keras.optimizers.rmsprop(lr=0.0005,decay=1e-6)
    model.compile(loss='categorical_crossentropy',
            optimizer=opt_rms,
            metrics=['accuracy'])
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)
    scores2 = model.evaluate_generator(
        test_generator, 
        steps=10)
    # Run 3
    # With optimization, compilation, and generation
    opt_rms = keras.optimizers.rmsprop(lr=0.0003,decay=1e-6)
    model.compile(loss='categorical_crossentropy',
            optimizer=opt_rms,
            metrics=['accuracy'])
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)
    model.save_weights(model_name + '_ep' + epochs*5 + '.h5')




    # Testin with test folder data
    score = model.evaluate_generator(
        test_generator, 
        steps=10)
    print ""
    return score

#=================================================================================================================================
# Generate parameters for all runs

def parameter_generator(past_parameters, past_results):
    parameters = []                                                     # Order: basemapnum
    if runs == 0:
        parameters = dict( activation1 = 'relu', activation2 = 'relu', activation3 = 'relu', 
                           activation4 = 'relu', activation5 = 'relu', activation6 = 'relu', 
                           decay1 = 1e-6,        decay2 = 1e-6,        decay3 = 1e-6,
                           decay4 = 1e-6,        decay5 = 1e-6,        decay6 = 1e-6,
                           dropout1 = 0.3,       dropout2 = 0.4,       dropout3 = 0.5,
                           epochs = 1,           batch_size = 4,       baseMapNum=32        )
    else:
        if scores[-1][1] > scores[-2]:
            parameters = past_parameters[-1]
        elif scores[-2] > scores[-1]:
            parameters = past_parameters[-2]
    change = random.choice(parameters.keys())
    parameters[change] = random.choice(param_grid[change])
    past_params.append(parameters)
    current_params = parameters
    return parameters



def run():
    #while True:
    parameters = parameter_generator(past_params, past_scores)
    print parameters
    model = model_generator(parameters)
    score = run_model(model, batch_size=4)
    print score
    past_parameters.append(parameters)
    past_scores.append(score)

run()


