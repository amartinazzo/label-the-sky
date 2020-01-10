
'''
exp00:

compare raw images (32x32xn features) to catalog (12+1 features)

01. build dataset
02. train log reg
03. predict on validation split
04. compute error
05. plot acc x magnitude curves (save as svg?)

'''

from keras.utils import to_categorical
import os
import numpy as np
import pandas as pd
from skimage import io
import sys


class_map = {'GALAXY': 0, 'STAR': 1, 'QSO': 2}


def build_flattened_dataset(csv_file, input_dir, nbands):
    df_orig = pd.read_csv(csv_file)
    df_orig = df_orig[(df_orig.photoflag==0)&(df_orig.ndet==12)]

    ext = '.png' if nbands==3 else '.npy'

    X = {}
    y = {}

    for split in ['train', 'val', 'test']:
        print('processing', split)
        df = df_orig[df_orig.split==split]

        y[split] = to_categorical(df['class'].apply(lambda c: class_map[c]).values, num_classes=3)

        img_list = df.id.values
        img_list = [os.path.join(input_dir, file.split('.')[0], file+ext) for file in img_list]

        if nbands == 3:
            read_func =  io.imread
        else:
            read_func = np.load

        img_shape = read_func(img_list[0]).shape

        X[split] = np.zeros((len(img_list), img_shape[0]*img_shape[1]*img_shape[2]))

        for ix, img in enumerate(img_list):
            x = read_func(img)
            x = np.flatten(x)
            X[split][ix,:] = x
        print('final shape', X[split].shape)

    return X, y


def build_catalog_dataset(csv_file):
    X = {}
    y = {}
    return X, y


########
# MAIN #
########


if __name__ == '__main__':

    if len(sys.argv) != 4:
        print('usage: python %s <data_dir> <csv_file> <nbands>' % sys.argv[0])
        exit(1)

    data_dir = sys.argv[1]
    csv_file = sys.argv[2]
    n_bands = int(sys.argv[3])

    # build datasets
    X, y = build_flattened_dataset(csv_file, data_dir, n_bands)
    # X_catalog, y_catalog = build_catalog_dataset()

    # 