import numpy as np
from pandas import read_csv
import keras


class_map = {'GALAXY': 0, 'STAR': 1, 'QSO': 2}


def get_sets(filepath, filters=None):
    """
    receives:
    * filepath      path to csv file with id values and class strings
    * filters       dict with optional min-max values, e.g. {'feature1': [min1, max1], 'feature2': [min2, max2]}

    returns: (X, y, labels) triplet, where
    * X is a list of object ids
    * y are integer-valued labels
    * labels is a dict mapping each id to its label, e.g. {'x1': 0, 'x2': 1, ...}
    """
    df = read_csv(filepath)
    print('original set size', df.shape)
    if filters is not None:
        for key, val in filters.items():
            df = df[df[key].between(val[0], val[1])]
        print('set size after filters', df.shape)
    X = df['id'].values
    y = df['class'].apply(lambda c: class_map[c]).values
    print(df['class'].value_counts(normalize=True))
    labels = dict(zip(X, y))

    return X, y, labels


class DataGenerator(keras.utils.Sequence):
    # adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    def __init__(
        self, object_ids, data_folder, labels=None, mode='classifier', 
        bands=None, batch_size=128, dim=(5500,1), n_classes=3, shuffle=True, extension='npy'):
        self.bands = bands
        self.batch_size = batch_size
        self.data_folder = data_folder
        self.extension = extension
        self.labels = labels
        self.mode = mode
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
        return np.maximum(len(self.object_ids) // self.batch_size, 1)


    def __getitem__(self, index):
        # generate indexes of one data batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_ids_temp = [self.object_ids[k] for k in indexes]
        X, y = self.__data_generation(list_ids_temp)

        if self.mode=='autoencoder':
            return X, X

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
        X = np.empty((self.batch_size,)+self.shape)
        if self.labels is not None:
            y = np.empty((self.batch_size), dtype=int)

        for i, object_id in enumerate(list_ids_temp):
            if self.extension == 'txt':
                spec = np.loadtxt(self.data_folder + object_id + '.txt').reshape(self.shape)
                spec = spec + 30000 # min = -30000
                spec = spec / 43000 # max = 13000; interval = 13000 - (-30000) = 43000
                X[i,] = spec
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
