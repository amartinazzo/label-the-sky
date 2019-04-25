import numpy as np
import keras

# adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

class DataGenerator(keras.utils.Sequence):
    def __init__(self, object_ids, labels, data_folder, batch_size=32, dim=(5500,1), n_classes=3, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.object_ids = object_ids
        self.data_folder = data_folder
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()


    def __len__(self):
        # define the number of batches per epoch
        return int(np.floor(len(self.object_ids) / self.batch_size))


    def __getitem__(self, index):
        # generate indexes of one data batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_ids_temp = [self.object_ids[k] for k in indexes]
        X, y = self.__data_generation(list_ids_temp)

        return X, y


    def on_epoch_end(self):
        # update indexes after each epoch
        self.indexes = np.arange(len(self.object_ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, list_ids_temp):
        # generate data containing batch_size samples
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        for i, object_id in enumerate(list_ids_temp):
            X[i,] = np.loadtxt(self.data_folder + object_id + '.txt').reshape(self.dim)
            y[i] = self.labels[object_id]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)