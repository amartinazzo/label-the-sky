'''
exp02:

01. load extracted features from val set
02. load 2d projections from val set
03. run clustering
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
    X_umap = np.load(f'{model_path}_X_{split}_features_umap.npy')    
    y = np.load(f'{model_path}_y_{split}.npy')
    y_hat = np.load(f'{model_path}_y_{split}_hat.npy')

    return X_feats, X_umap, y, y_hat


########
# MAIN #
########


if __name__ == '__main__':
    set_random_seeds()

    if len(sys.argv) < 2:
        print('usage: python %s <model_file> <split: optional>' % sys.argv[0])
        exit(1)

    model_file = sys.argv[1]
    split = sys.argv[2] if len(sys.argv)==3 else 'val'

    results_folder = os.getenv('HOME')+'/label_the_sky/results'
    model_path = os.path.join(results_folder, model_file)

    X, X_umap, y, y_hat = load_features(model_path, split)

    y_labels = {}

    # k-means varying k
    # TODO silhouette
    for k in range(3, 8):
        print('k-means; k=', k)
        kmeans = KMeans(n_clusters=k, n_init=10, n_jobs=-1).fit(X)
        y_labels[f'kmeans_{k}'] = kmeans.labels_

    # hdbscan
    # hdbscan = HDBSCAN(n_jobs=-1).fit(X)
    # y_labels['hdbscan'] = hdbscan.labels_
