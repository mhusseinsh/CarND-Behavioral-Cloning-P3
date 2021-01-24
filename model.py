from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers import Lambda, Cropping2D
from keras.layers.pooling import MaxPooling2D
import numpy as np
import json
import visualkeras


class Model:
    def __init__(self, img_shape, top_crop, bottom_crop, batch_size, epochs, loss, optimizer, verbose, train_generator,
                 validation_generator,
                 train_samples, validation_samples, model_name):
        self.img_shape = img_shape
        self.epochs = epochs
        self.loss = loss
        self.optimizer = optimizer
        self.verbose = verbose
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        self.train_samples = train_samples
        self.validation_samples = validation_samples
        self.batch_size = batch_size
        self.model_name = model_name
        self.top_crop = top_crop
        self.bottom_crop = bottom_crop

    def create_model(self):
        model = Sequential()
        # normalization
        model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=self.img_shape))

        # crops at top and bottom
        model.add(Cropping2D(cropping=((self.top_crop, self.bottom_crop), (0, 0))))

        # convolutional layers
        model.add(Convolution2D(24, (5, 5), strides=(2, 2), activation='relu'))
        model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation='relu'))
        model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation='relu'))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(Convolution2D(64, (3, 3), activation='relu'))

        # flattening
        model.add(Flatten())

        # fully connected layers with dropouts
        model.add(Dense(100))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(50))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(10))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1))

        return model

    def train_model(self, model):
        # Compiling and training the model
        print("Training Model with: ")
        print('\tNumber of Epochs: {}'.format(self.epochs))
        print('\tBatch Size: {}'.format(self.batch_size))
        print('\tTraining Loss: {}'.format(self.loss))
        print('\tOptimizer: {}'.format(self.optimizer))
        model.compile(loss=self.loss, optimizer=self.optimizer)
        history_object = model.fit_generator(self.train_generator,
                                             steps_per_epoch=np.ceil(len(self.train_samples) / self.batch_size),
                                             validation_data=self.validation_generator,
                                             validation_steps=np.ceil(len(self.validation_samples) / self.batch_size),
                                             epochs=self.epochs,
                                             verbose=self.verbose)

        model.save(self.model_name + '.h5')
        print('Model saved as {}.h5'.format(self.model_name))
        return history_object

    def get_summary(self, model):
        model.summary()

    def save_model(self, model):
        ################################################################
        # Save the model and weights
        #################################################################
        model_json = model.to_json()
        with open(self.model_name + ".json", "w") as json_file:
            json.dump(model_json, json_file)
        model.save_weights(self.model_name + '.h5')
        print("Saved model to disk")
