"""
This file contains convolutional neural network based simple model which is trained from scratch
Model Details:

"""
import os
import keras
import numpy as np
from random import shuffle
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

import PBModel.config as config
from PBModel.model.dataGenerator import DataGeneratorClass


class CustomDataGeneratorClass:
    """
    Creates training and validation data generator when data is arranged as following:
    Data\
        Positive\
            Source1\
                img1
                img2
                .
                .
            Source2\
            . 
            .
        Negative\
            Source1\
            Source2\
            . 
            .
    Note: Location of positive data is picked from config.RAW_DATA_POSITIVE and 
    similarily for Negative data
    """

    def __init__(self, batch_size=32, dim=(32,32), n_channels=3, n_classes=2, shuffle=True, debugMode=False):
        """
        Assign ids to all images. Also creates mapping from id to label and address 
        Creates training and validation data generators
        @params:
            debugMode: When true, uses less data
        """
        self.debugMode = debugMode
        self.generateLabelsAndIds()
        self.trainingGenerator   = DataGeneratorClass(
            self.ids["train"], self.labels, self.address,
            batch_size, dim, n_channels, n_classes, shuffle
        )
        self.validationGenerator = DataGeneratorClass(
            self.ids["validation"], self.labels, self.address,
            batch_size, dim, n_channels, n_classes, shuffle
        ) 
    
    def generateLabelsAndIds(self):
        """
        ids: {
           train: [] # list of ids of images in train
           validation: [] # list of ids of iamges in validation
        }
        labels: Given id, provide label of image
        address: Given id, provide address for image
        """
        rawIds = [] 
        self.address = {}
        self.labels = {}

        posImgFolder = config.ROOT + config.RAW_DATA_NEGATIVE
        negImgFolder = config.ROOT + config.RAW_DATA_POSITIVE
        id = 0
        for root, dirs, files in os.walk(posImgFolder):
            for file in files: 
                rawIds.append(id)
                self.address[id] = os.path.join(root, file)
                self.labels[id] = 1.0
                id+=1

        for root, dirs, files in os.walk(negImgFolder):
            for file in files: 
                rawIds.append(id)
                self.address[id] = os.path.join(root, file)
                self.labels[id] = 0.0
                id+=1
        
        shuffle(rawIds) # XXX Current will not ensure equal positive negative presence

        if self.debugMode: 
            rawIds = rawIds[0: min(1000, len(rawIds))]

        self.ids = {}
        trainSize = int(np.floor(len(rawIds)*0.8))
        self.ids["train"] = rawIds[:trainSize]
        self.ids["validation"] = rawIds[trainSize:]

class CustomModelClass:
    """
    This class contains information about model and datasource for training
    """

    def __init__(self, debugMode = False):
        self.img_width  = 32 
        self.img_height = 32 
        self.dataGenerator = CustomDataGeneratorClass(debugMode = debugMode)
        self.getModel()

    def getModel(self):
        if keras.backend.image_data_format() == 'channels_first':
            input_shape = (3, self.img_width, self.img_height)
        else:
            input_shape = (self.img_width, self.img_height, 3)

        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])

        self.model = model

    def train(self):
        """
        Train model on the dataset
        """
        sE = len(self.dataGenerator.ids["train"])// 32
        sV = len(self.dataGenerator.ids["validation"])// 32
        self.model.fit_generator(
            generator=self.dataGenerator.trainingGenerator,
            steps_per_epoch= sE,
            epochs=10,
            validation_data=self.dataGenerator.validationGenerator,
            validation_steps=sV,
            use_multiprocessing=True,
            workers=2,
        )
    
if __name__ == "__main__":
    myModel = CustomModelClass(debugMode=True)
    myModel.train()