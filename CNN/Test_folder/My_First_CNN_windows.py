# Rohan Borkar
# Neural network
# For big image datasets. Like 10k+ samples.
# Unfortunately you have to run this with python 3 because windows only has keras on p3. Stupid windows...
# Import keras
import keras
from keras.models import Sequential
from keras import backend as K
# Preprocessing. AKA loading images to use
from keras.preprocessing.image import ImageDataGenerator
from keras.backend import image_data_format
# Layer and optimization
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
# Numpy is bae <3
import numpy as np
import random
import os.path
import pandas as pd

#=================================================================================================================================D
# The following are parameters for the import data and model. I like to keep them at the top

model_name = 'NHIcxr'
num_classes = 15                                                            # Number of classes should the model should distinguish
img_width, img_height = 1024, 1024                                          # Width and height of images in dataset
img_shape = (1024, 1024, 3)                                                 # Shape of the image (width, height, depth)
nb_train_samples = 8                                                        # How many samples to use in training per step
nb_validation_samples = 4                                                   # How many samples to use in validation per step


# Data directories and details
train_data_dir = 'C:data/train'                                             # Location of training data (for training the model)
validation_data_dir = 'C:data/validation'                                   # Location of validation data (for estimating training quality)
test_data_dir = 'C:data/test'                                               # Location of testing data (for application)


# Hyperparameters
activations = ['relu', 'tanh', 'sigmoid']                                   # What kind of curve to use for activation layers
weight_decays = [1e-6, 1e-4, 1e-2, 1, 10]                                   # I'm so confused
optimizers = ['rmsprop', 'adam', 'adagrad']                                 # Optimizers to optimize...stuff. I really have no idea
#batch_sizes = [16, 32, 64, 128]                                            # Higher the better
batch_sizes = [2]                                                           # Low batch number for testing purposes only
dropouts = [0.2, 0.3, 0.4, 0.5, 0.6]                                        # I dunno man, Rah says it makes the model "turn off its brain"
#epochs = [20, 25, 30, 40]                                                  # How many iterations to do. This is multiplied by 5
epochs = [1, 2]                                                             # Low number of epochs for testing purposes only
learning_rates = [0.01, 0.1, 0.05, 0.001]                                   # How drastically to edit weights when model guesses wrong
baseMapNums = [32, 64, 128, 256, 512, 1024]                                 # No idea what this is loool. Used as 'filter' in Conv2D

param_grid = dict( activation1 = activations, activation2 = activations, activation3 = activations, 
                   activation4 = activations, activation5 = activations, activation6 = activations, 
                   decay = weight_decays,     optimizer = optimizers,    batch_size = batch_sizes,
                   dropout1 = dropouts,       dropout2 = dropouts,       dropout3 = dropouts,
                   epochs = epochs,           lr1 = learning_rates,      baseMapNum = baseMapNums   )



past_params = None
current_run = 0
if os.path.isfile("C:" + model_name + "_results.csv"):
    past_params = pd.read_csv("C:" + model_name + "_results.csv")
    current_run = past_params['run'].max() + 1
    past_params.drop(['run', 'accuracy', 'loss'], axis=1)
else:
    past_params = pd.DataFrame(columns = ['activation1', 'activation2', 'activation3',
                                          'activation4', 'activation5', 'activation6',
                                          'decay',       'optimizer',   'batch_size',
                                          'dropout1',    'dropout2',    'dropout3',
                                          'epochs',      'lr1',         'baseMapNum'  ])
 
#=================================================================================================================================D
# Here are functions that produce/run models and parameters. Want me to explain this shit? Too bad, I can't. Get shrekt, m9

# Give model_generator will make a model with specific hyperparameters
def model_generator( param_dict ):

    model = Sequential()
    model.add(Conv2D(param_dict['baseMapNum'], (3,3), padding='same', kernel_regularizer=regularizers.l2(param_dict['decay']), input_shape=img_shape))
    model.add(Activation(param_dict['activation1']))
    model.add(BatchNormalization())
    model.add(Conv2D(param_dict['baseMapNum'], (3,3), padding='same', kernel_regularizer=regularizers.l2(param_dict['decay'])))
    model.add(Activation(param_dict['activation2']))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(param_dict['dropout1']))

    model.add(Conv2D(2*param_dict['baseMapNum'], (3,3), padding='same', kernel_regularizer=regularizers.l2(param_dict['decay'])))
    model.add(Activation(param_dict['activation3']))
    model.add(BatchNormalization())
    model.add(Conv2D(2*param_dict['baseMapNum'], (3,3), padding='same', kernel_regularizer=regularizers.l2(param_dict['decay'])))
    model.add(Activation(param_dict['activation4']))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(param_dict['dropout2']))

    model.add(Conv2D(4*param_dict['baseMapNum'], (3,3), padding='same', kernel_regularizer=regularizers.l2(param_dict['decay'])))
    model.add(Activation(param_dict['activation5']))
    model.add(BatchNormalization())
    model.add(Conv2D(4*param_dict['baseMapNum'], (3,3), padding='same', kernel_regularizer=regularizers.l2(param_dict['decay'])))
    model.add(Activation(param_dict['activation6']))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(param_dict['dropout3']))

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    return model

#=================================================================================================================================D
# Running the model. Man, I don't understand any of this lol
# Designed to run from a folder of images
    # Augmentation configuration we will use for training.
def run_model(model, param_dict):
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
        batch_size=param_dict['batch_size'],
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=param_dict['batch_size'],
        class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=param_dict['batch_size'],
        class_mode='categorical')


    # Training
    # Run 1
    # With optimization, compilation, and generation
    opt_rms = keras.optimizers.rmsprop(lr=param_dict['lr1'],decay=1e-6)
    model.compile(loss='categorical_crossentropy',
            optimizer=opt_rms,
            metrics=['accuracy'])
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // param_dict['batch_size'],
        epochs=param_dict['epochs']*3,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // param_dict['batch_size'])

    # Run 2
    # With optimization, compilation, and generation
    opt_rms = keras.optimizers.rmsprop(lr=param_dict['lr1']/2,decay=1e-6)
    model.compile(loss='categorical_crossentropy',
            optimizer=opt_rms,
            metrics=['accuracy'])
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // param_dict['batch_size'],
        epochs=param_dict['epochs'],
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // param_dict['batch_size'])

    # Run 3
    # With optimization, compilation, and generation
    opt_rms = keras.optimizers.rmsprop(lr=param_dict['lr1']/3,decay=1e-6)
    model.compile(loss='categorical_crossentropy',
            optimizer=opt_rms,
            metrics=['accuracy'])
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // param_dict['batch_size'],
        epochs=param_dict['epochs'],
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // param_dict['batch_size'])
    model.save_weights(model_name + '_ep' + str(epochs*5) + '.h5')




    # Testin with test folder data
    score = model.evaluate_generator(
        test_generator, 
        steps=10)
    print (score)
    return score

#=================================================================================================================================

# Generate parameters for all runs. I'm just a monkey with a typewriter, man.

def parameter_generator():
    global past_params
    parameters = dict()
    # If first run, do default parameters
    if current_run == 0:
        parameters = dict( activation1 = 'relu', activation2 = 'relu', activation3 = 'relu', 
                           activation4 = 'relu', activation5 = 'relu', activation6 = 'relu', 
                           decay = 1e-6,         optimizer = 'rmsprop',batch_size = 2,        #Change this when deploy on server
                           dropout1 = 0.3,       dropout2 = 0.4,       dropout3 = 0.5,
                           epochs = 1,           lr1 = 0.001,          baseMapNum=2        )


    # NEEDS EDITS
    # If at a run divisible by 10, use the best score and go from there
    elif (current_run%20) == 0:
        while True:
            for col in past_params.columns:
                parameters[col] = 
            try:
                parameters = parameters.pop('Unnamed: 0')
                print (parameters.type())
            except:
                print (parameters.type())

            hyperparameter = random.choice(list(parameters.keys()))
            parameters[hyperparameter] = random.choice(list(param_grid[hyperparameter]))
            if not (past_params == parameters).all():
                break

    # Otherwise just randomise one parameter
    else:
        while True:
            parameters = past_params.tail(1).to_dict(orient='records')
            try:
                parameters = parameters.pop('Unnamed: 0')
                print (parameters.type())
            except:
                print (parameters.type())
            hyperparameter = random.choice(list(parameters.keys()))
            parameters[hyperparameter] = random.choice(list(param_grid[hyperparameter]))
            if not (past_params == parameters).all():
                break
    return parameters





def run():
    # Generate parameters. Uses randomization
    print ("\nCreating parameters...\n")
    param_dict = parameter_generator()
    # Make model based on parameters
    print ("\nCreating model...\n")
    model = model_generator(param_dict)
    # Run model with parameters
    print ("\nRunning model...\n")
    #score = run_model(model, param_dict)
    score = [566, 0.1] # Made up results for testing
    print ("\nRecording results...\n")
    # Add parameters + results to past_params
    global current_run
    results = dict ( loss = score[0], accuracy = score[1], run = current_run )
    param_dict.update( results )
    # Now that we have results, append the parameters we used plus the score they got to the results dataframe
    new_row = pd.DataFrame.from_dict([param_dict], orient='columns')
    global past_params
    current_run += 1
    if os.path.isfile("C:" + model_name + "_results.csv"):
        saved_params = pd.read_csv("C:" + model_name + "_results.csv")
        saved_params = pd.concat([saved_params, new_row])
        saved_params.to_csv("C:" + model_name + "_results.csv")
    else:
        new_row.set_index('run')
        new_row.to_csv("C:" + model_name + "_results.csv")
    print ("Run " + str(current_run) + " completed with " + str(results['accuracy']*100) + " percent accuracy")


for i in range(50):
    run()



"""
To do:
Add dataframe + csv support
Add optimizer to run_model

"""
