from classifiers.models import resnext
from datagen import DataGenerator
from glob import glob
# from MulticoreTSNE import MulticoreTSNE as TSNE
import numpy as np
import os
import pandas as pd
# from hdbscan import HDBSCAN
from time import time
from umap import UMAP
from utils import get_sets


#########################
# BEGIN PARAMETER SETUP #
#########################

cardinality = 4
depth = 11
n_epoch = 500
width = 16
pooling = 'avg'

task = 'regression' # classification or regression (magnitudes)
csv_dataset = 'csv/dr1_classes_mag1418_split_ndet.csv'
weights_file = f'classifiers/image-models/depth{depth}_card{cardinality}_eph{n_epoch}_{task}_mag1418.h5'

run_clustering = False
run_tsne = False
run_umap = True

#######################
# END PARAMETER SETUP #
#######################

filename = f'{pooling}pool_'+weights_file.split('/')[-1][:-3]
features_file = f'npy/features_{filename}.npy'
tsne_file = f'npy/tsne_maxpool_{filename}.npy'
umap_file = f'npy/umap_maxpool_{filename}.npy'

img_dim = (32,32,12)
n_classes = 3 if task=='magnitudes' else 12
data_mode = 'classes' if task=='classification' else 'magnitudes'
lst_activation = 'softmax' if task=='classification' else 'linear'


# extract features
if not os.path.exists(features_file):
	data_path = os.environ['DATA_PATH']
	params = {'data_folder': data_path+'/crops_asinh/', 'dim': img_dim, 'n_classes': n_classes}

	os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
	os.environ['CUDA_VISIBLE_DEVICES']='0'

	# create model
	model = resnext(img_dim, depth=depth, cardinality=cardinality, width=width, classes=n_classes,
		top_layer=False, pooling=pooling)
	print('model created')
	model.load_weights(weights_file, by_name=True, skip_mismatch=True)
	print('model weights loaded')

	# gen dataset
	df = pd.read_csv(csv_dataset)
	df = df[(df.split!='test') & (df.n_det==12) & (~df['class'].isna())]
	print('df shape', df.shape)
	X, y_true, _ = get_sets(df, mode=data_mode)
	X_generator = DataGenerator(X, batch_size=1, shuffle=False, **params)

	print('extracting features')
	X_features = model.predict_generator(X_generator, steps=len(X), verbose=0)

	# dtype is float32. each float32 = 4 bytes.
	# total size of X_features matrix will be:
	# 117584 x 512 x 4 ~= 230 MiB
	print('X_features shape', X_features.shape)
	print('X_features.dtype', X_features.dtype)

	np.save(features_file, X_features)
	print('saved ', features_file)

else:
	X_features = np.load(features_file)


# cluster
if run_clustering:
	start = time()
	clustering = HDBSCAN(core_dist_n_jobs=-1).fit(X_features)
	print('minutes taken:', int((time()-start)/60))

	y_cluster = clustering.labels_
	np.save(clusters_file, y_cluster)
	print(pd.Series(y_cluster).value_counts(normalize=True))
	print(pd.Series(y_true).value_counts(normalize=True))


# generate 2d embeddings
if run_umap:
	print('embedding with UMAP')
	start = time()
	X_umap = UMAP().fit_transform(X_features)
	np.save(umap_file, X_umap)
	print('minutes taken:', int((time()-start)/60))

if run_tsne:
	print('embedding with T-SNE')
	start = time()
	X_tsne = TSNE(n_jobs=-1).fit_transform(X_features)
	np.save(tsne_file, X_tsne)
	print('minutes taken:', int((time()-start)/60))
