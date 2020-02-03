'''
exp02:

01. load extracted features from val set
02. load 2d projections from val set
03. run clustering (k-means, hbdscan)
04. plot clusters in 2d

'''

from _exp01 import set_random_seeds
from hdbscan import HDBSCAN
import numpy as np
from sklearn.cluster import k_means
from time import time
from umap import UMAP


def load_features(model_path, split='val'):
	X_feats = np.load(f'{model_path}_X_{split}_features.npy')
	y = np.load(f'{model_path}_y_{split}.npy')
	y_hat = np.load(f'{model_path}_y_{split}_hat.npy')

	return X_feats, y, y_hat


########
# MAIN #
########


if __name__ == '__main__':
    set_random_seeds()