from albumentations import Compose, Flip, HorizontalFlip, RandomRotate90, ShiftScaleRotate
from cv2 import imread
import numpy as np
import keras


class DataGenerator(keras.utils.Sequence):
    # adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    def __init__(
        self, object_ids, data_folder, n_bands, target='classes', labels=None, 
        batch_size=32, dim=(5500,1), n_classes=3, shuffle=True, extension='npy', augmentation=True):
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
        self.augmentation = augmentation
        self.aug = self.compose_augment()

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

        X = np.float32(X)

        if self.mode=='autoencoder':
            return X, X

        if y is None:
            return X

        return X, y


    def compose_augment(self, p=.9):
        return Compose([
            Flip(),
            HorizontalFlip(),
            # ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=.5),
        ], p=p)


    def augment(self, image):
        return self.aug(image=image)['image']


    def on_epoch_end(self):
        # update indexes after each epoch
        self.indexes = np.arange(len(self.object_ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, list_ids_temp):
        # generate data containing batch_size samples
        X = np.empty((self.batch_size,)+self.shape)
        if self.mode=='classes':
            y = np.empty((self.batch_size), dtype=int)
        elif self.mode=='magnitudes':
            y = np.zeros((self.batch_size, self.n_classes), dtype=float)

        for i, object_id in enumerate(list_ids_temp):
            if self.extension == 'txt':
                spec = np.loadtxt(self.data_folder + object_id + '.txt').reshape(self.shape)
                spec = spec + 30000 # min = -30000
                spec = spec / 43000 # max = 13000; interval = 13000 - (-30000) = 43000
                X[i,] = spec
            elif self.extension == 'png':
                im = imread(self.data_folder + object_id.split('.')[0] + '/' + object_id + '.png')
                if self.augmentation:
                    im = self.augment(im)
                X[i,] = im
            else:
                if self.bands is not None:
                    arr = np.load(self.data_folder + object_id.split('.')[0] + '/' + object_id + '.npy').reshape(self.shape_orig)
                    arr = arr[:,:,self.bands]
                    X[i,] = arr
                else:
                    im = np.load(self.data_folder + object_id.split('.')[0] + '/' + object_id + '.npy').reshape(self.shape)
                    if self.augmentation:
                        im = self.augment(im)
                    X[i,] = im
            if self.labels is not None:
                y[i,] = np.array(self.labels[object_id])

        if self.labels is None:
            return X, None
        elif self.mode=='magnitudes':
            return X, y

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
