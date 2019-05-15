import numpy as np
import keras

# adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

class DataGenerator(keras.utils.Sequence):
    def __init__(
        self, object_ids, data_folder, labels=None, bands=None, batch_size=256, dim=(5500,1), n_classes=3, shuffle=True, extension='npy'):
        self.bands = bands
        self.batch_size = batch_size
        self.data_folder = data_folder
        self.extension = extension
        self.labels = labels
        self.n_classes = n_classes
        self.object_ids = object_ids
        self.shape_orig = dim
        self.shape = dim
        self.shuffle = shuffle

        if bands is not None:
            self.shape = dim[:-1] + (len(bands),)

        self.on_epoch_end()


    def __len__(self):
        # define the number of batches per epoch
        return int(np.floor(len(self.object_ids) / self.batch_size))


    def __getitem__(self, index):
        # generate indexes of one data batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_ids_temp = [self.object_ids[k] for k in indexes]
        X, y = self.__data_generation(list_ids_temp)

        if y is None:
            return X

        return X, y


    def on_epoch_end(self):
        # update indexes after each epoch
        self.indexes = np.arange(len(self.object_ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, list_ids_temp):
        # generate data containing batch_size samples
        X = np.empty((self.batch_size, *self.shape))
        if self.labels is not None:
            y = np.empty((self.batch_size), dtype=int)

        for i, object_id in enumerate(list_ids_temp):
            if self.extension == 'txt':
                X[i,] = np.loadtxt(self.data_folder + object_id + '.txt').reshape(self.shape)
            else:
                if self.bands is not None:
                    arr = np.load(self.data_folder + object_id + '.npy').reshape(self.shape_orig)
                    arr = arr[:,:,self.bands]
                    X[i,] = arr
                else:
                    X[i,] = np.load(self.data_folder + object_id + '.npy').reshape(self.shape)
            if self.labels is not None:
                y[i] = self.labels[object_id]

        if self.labels is None:
            return X, None

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)