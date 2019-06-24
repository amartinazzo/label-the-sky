from classifiers.models import resnext
from datagen import DataGenerator
from glob import glob
from keras.optimizers import Adam
from MulticoreTSNE import MulticoreTSNE as TSNE
import numpy as np
import os
import pandas as pd
from hdbscan import HDBSCAN
from sklearn.preprocessing import MinMaxScaler
from time import time
from umap import UMAP
from utils import get_sets


clusters_file = 'dr1_diff_dbscan.npy'
csv_file = 'csv/diff_cat_dr1.csv' #'csv/matched_cat_dr1.csv'
features_file = 'dr1_diff_features.npy'
tsne_file = 'dr1_diff_features_tsne.npy'
umap_file = 'dr1_diff_features_umap.npy'
run_clustering = False
run_embeddings = True

# extract features

if not os.path.exists(features_file):
	batch_size = 256
	cardinality = 4
	depth = 29
	img_dim = (32,32,12)
	n_classes = 3
	width = 16

	models_dir = 'classifiers/image-models/'
	weights_file = 'classifiers/image-models/resnext_depth29_card4_300epc_weights.h5'
	home_path = os.path.expanduser('~')
	params = {'data_folder': home_path+'/raw-data/crops/unsup_normalized/', 'dim': img_dim, 'n_classes': n_classes}
	class_weights = {0: 1, 1: 1.25, 2: 5} # 1/class_proportion

	os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
	os.environ['CUDA_VISIBLE_DEVICES']='0'

	# create resnext model
	model = resnext(img_dim, depth=depth, cardinality=cardinality, width=width, classes=n_classes,
		include_top=False, pooling='max')
	print('model created')

	model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])
	print('finished compiling')

	model.load_weights(weights_file, by_name=True, skip_mismatch=True)
	print('model weights loaded!')

	obj_list = glob(home_path+'/raw-data/crops/unsup_normalized/*', recursive=True)
	obj_list = [s.split('/')[-1][:-4] for s in obj_list]
	X, y_true, _ = get_sets(csv_file, {'fwhm': [0,32]}, obj_list)
	X_generator = DataGenerator(X, batch_size=1, shuffle=False, **params)
	print('extracting features')
	X_features = model.predict_generator(X_generator, steps=len(X), verbose=1)

	# dtype is float32. each float32 = 4 bytes.
	# total size of X_features matrix will be:
	# 117584 x 512 x 4 ~= 230 MiB
	print(X_features.shape)
	print(X_features.dtype)

	np.save(features_file, X_features)
	print('saved feature matrix')

else:
	_, y_true, _ = get_sets(csv_file)
	X_features = np.load(features_file)

print('features max', X_features.max())
print('features min', X_features.min())
print('y_true shape', y_true.shape)
print('features shape', X_features.shape)

# cluster

if run_clustering:
	start = time()
	scaler = MinMaxScaler()
	X_features = scaler.fit_transform(X_features)
	clustering = HDBSCAN(core_dist_n_jobs=-1).fit(X_features)
	print('minutes taken:', int((time()-start)/60))

	y_cluster = clustering.labels_
	np.save(clusters_file, y_cluster)
	print(pd.Series(y_cluster).value_counts(normalize=True))
	print(pd.Series(y_true).value_counts(normalize=True))

# generate 2d embeddings

if run_embeddings:
	print('embedding with UMAP')
	start = time()
	X_umap = UMAP().fit_transform(X_features)
	np.save(umap_file, X_umap)
	print('minutes taken:', int((time()-start)/60))
	print('embedding with T-SNE')
	start = time()
	X_tsne = TSNE(n_jobs=-1).fit_transform(X_features)
	np.save(tsne_file, X_tsne)
	print('minutes taken:', int((time()-start)/60))