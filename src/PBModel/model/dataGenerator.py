import os
import keras
import numpy as np

import PBModel.config as config
import PBModel.model.utils as modelUtils

class DataGeneratorClass(keras.utils.Sequence):
    """
    Base data generator class
    """
    def __init__(self, list_IDs, labels, address, batch_size=32, dim=(32,32), n_channels=3,
                 n_classes=2, shuffle=True):
        """
        Initialization
        @params:
            list_IDs    : list of ids of all images (entire dataset)
            labels      : dictionary mapping id => label
            address     : dictionary mapping id => address of image on disk
            batch_size  : 
            dim         : dimension of image to pass in model
            n_channels  : 
            n_classes   :
            shuffle     : shuffle data after each epoch (TODO)
        """
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.address = address
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, list_IDs_temp):
        """
        Generates data containing batch_size samples
        """
        X = np.empty(( (self.batch_size,) + self.dim + (self.n_channels, )))
        y = np.empty((self.batch_size), dtype=int)
        for i, ID in enumerate(list_IDs_temp):
            try:
                X[i,] = modelUtils.loadImage(self.address[ID], self.dim)
            except Exception as e:
                print "Exception Thrown while loading image ", self.address[ID]
                print e 
                continue
            y[i] = self.labels[ID]
        return X, y