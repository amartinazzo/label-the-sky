from _datagen import get_dataset
from pandas import read_csv
import numpy as np
import os
from skimage import io
import sys


if len(sys.argv) != 2:
    print('usage: python {} <n_channels>'.format(
        sys.argv[0]))
    exit(1)

n_channels = int(sys.argv[1])

if n_channels not in [3, 12]:
    raise ValueError('n_channels must be: 3, 12')

base_dir = os.environ['DATA_PATH']
data_dir = 'crops_rgb32' if n_channels==3 else 'crops_calib'
data_dir = os.path.join(base_dir, data_dir)
output_dir = os.path.join(os.environ['HOME'], 'data')

dtype = 'uint8' if n_channels==3 else 'float32'
read_fn = io.imread if n_channels==3 else np.load
ext = '.png' if n_channels==3 else '.npy'

df = read_csv('csv/dr1_split.csv')

for pretraining in [True, False]:
    pretraining_str = 'pretraining' if pretraining else 'clf'
    for split in ['train', 'val', 'test']:
        df_tmp = df[(df.pretraining==pretraining) & (df.split==split)]
        for target in ['class', 'magnitudes', 'mockedmagnitudes']:
            ids, y, labels = get_dataset(df_tmp, target=target, n_bands=n_channels)
            np.save(
                os.path.join(
                    output_dir,
                    f'{pretraining_str}_{n_channels}_y_{target}_{split}.npy'),
                y)
            print(f'saved {pretraining_str}_{n_channels}_y_{target}_{split}.npy', y.shape)
        im_paths = [os.path.join(data_dir, i.split('.')[0], i + ext) for i in ids]
        X = np.zeros((len(ids),) + (32, 32, n_channels), dtype=dtype)
        for i, path in enumerate(im_paths):
            X[i, :] = read_fn(path)
        np.save(
            os.path.join(
                output_dir,
                f'{pretraining_str}_{n_channels}_X_{split}.npy'),
            X)
        print(f'saved {pretraining_str}_{n_channels}_X_{split}.npy', X.shape)
