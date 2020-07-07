from albumentations import Compose, Flip, HorizontalFlip
from _base import MAG_MAX
from cv2 import imread
import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical
import os


CLASS_MAP = {'GALAXY': 0, 'STAR': 1, 'QSO': 2}


def get_dataset(df, target='classes', n_bands=12, filters=None):
    """
    receives:
    * df            pandas dataframe
    * filters       dict with optional min-max values,
                    e.g. {'feature1': [min1, max1], 'feature2': [min2, max2]}
    * obj_list      list of object ids (used to check existing files)
    * target        classes, magnitudes, mockedmagnitudes

    returns: (ids, y, labels) triplet, where
    * ids is a list of object ids
    * y are categorical labels
    * labels is a dict mapping each id to its label, e.g. {'x1': 0, 'x2': 1}
    """
    if filters is not None:
        for key, val in filters.items():
            df = df[df[key].between(val[0], val[1])]
    ids = df.id.values

    if target=='classes':
        y = df['class'].apply(lambda c: CLASS_MAP[c]).values
        y = to_categorical(y, num_classes=3)
    elif target=='magnitudes':
        if n_bands==5:
            y = df[['u','g','r','i','z']].values
        else:
            y = df[['u','f378','f395','f410','f430','g','f515','r','f660','i','f861','z']].values
        y = y/MAG_MAX
    elif target=='mockedmagnitudes':
        if n_bands==5:
            y = df[['u_mock','g_mock','r_mock','i_mock','z_mock']].values
        else:
            y = df[[
                'u_mock','f378_mock','f395_mock','f410_mock','f430_mock','g_mock',
                'f515_mock','r_mock','f660_mock','i_mock','f861_mock','z_mock']].values
        y = y/MAG_MAX
    elif target=='redshifts':
        y = df[['redshift_base', 'redshift_exp']].values
    else:
        return X, _, _

    labels = dict(zip(ids, y))
    return ids, y, labels


class DataGenerator(Sequence):
    """
    adapted from
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """

    def __init__(
            self, object_ids, data_folder, input_dim, target='classes',
            labels=None, batch_size=32, n_outputs=3, shuffle=True,
            augmentation=True):
        self.batch_size = batch_size
        self.data_folder = data_folder
        self.labels = labels
        self.n_outputs = n_outputs
        self.object_ids = object_ids
        self.shape = input_dim
        self.shuffle = shuffle
        self.target = target

        self.augmentation = augmentation
        self.bands = [0, 5, 7, 9, 11] if self.shape[2] == 5 else None
        self.aug = self.compose_augment()
        self.extension = '.npy' if self.shape[2] > 3 else '.png'
        self.shape_orig = self.shape[:-1] + (12,)

        self.on_epoch_end()

    def __len__(self):
        # define the number of batches per epoch
        return np.maximum(len(self.object_ids) // self.batch_size, 1)

    def __getitem__(self, index):
        # generate indexes of one data batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_ids_temp = [self.object_ids[k] for k in indexes]
        X, y = self.__data_generation(list_ids_temp)
        # X = np.float32(X)

        if self.target == 'autoencoder':
            return X, X

        if y is None:
            return X

        return X, y

    def on_epoch_end(self):
        # update indexes after each epoch
        self.indexes = np.arange(len(self.object_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def compose_augment(self, p=.9):
        return Compose([
            Flip(),
            HorizontalFlip(),
        ], p=p)

    def augment(self, image):
        return self.aug(image=image)['image']

    def __data_generation(self, list_ids_temp):
        # generate data containing batch_size samples
        X = np.empty((self.batch_size,) + self.shape)
        y = np.zeros((self.batch_size, self.n_outputs), dtype=np.float32)

        for i, object_id in enumerate(list_ids_temp):
            filepath = os.path.join(
                self.data_folder,
                object_id.split('.')[0],
                object_id + self.extension)

            if self.extension == '.png':
                im = imread(filepath)
            else:
                if self.bands is not None:
                    im = np.load(filepath).reshape(self.shape_orig)
                    im = im[:, :, self.bands]
                else:
                    im = np.load(filepath).reshape(self.shape)

            if self.augmentation:
                im = self.augment(im)
            X[i, :] = im

            if self.labels is not None:
                y[i, :] = self.labels[object_id]

        if self.labels is None:
            return X, None

        return X, y
