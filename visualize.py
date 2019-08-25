from astropy import visualization as vis
from astropy.io import fits
from astropy.visualization import make_lupton_rgb
import cv2
import matplotlib
from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerPathCollection
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd
from utils import get_sets


matplotlib.rcParams.update({'font.size': 6})

# filename = '../raw-data/unsup_normalized/SPLUS.STRIPE82-0001.00003.npy'


bands = ['U', 'F378', 'F395', 'F410', 'F430', 'G', 'F515', 'R', 'F660', 'I', 'F861', 'Z']

stretcher = vis.LogStretch(a=1e0)
scaler = vis.ZScaleInterval()


def plot_single_band(im_path):
	fits_im = fits.open(im_path)
	data = fits_im[1].data #ndarray
	print(data)
	print(data.max(), data.min())
	data_orig = data
	data = sqrt_stretcher(data)
	vmin, vmax = scaler.get_limits(data)
	plt.imshow(data, cmap='gray', vmin=vmin, vmax=vmax)
	plt.show()


def plot_field_rgb(filepath):
	fits_im = fits.open(filepath.format('R'))
	data_r = fits_im[1].data
	fits_im = fits.open(filepath.format('G'))
	data_g = fits_im[1].data
	fits_im = fits.open(filepath.format('U'))
	data_u = fits_im[1].data
	im = make_lupton_rgb(data_r, data_g, data_u, stretch=1)
	plt.imshow(im, interpolation='nearest')
	plt.axis('off')
	plt.show()


# receives 12-band array and plots each band in grayscale + lupton composite + given rgb composite
def plot_bands(filename, plot_composites=True, znorm=True, base_catalog='csv/splus_catalog_dr1.csv'):
	arr = np.load(filename)
	if znorm:
		arr = arr - np.mean(arr, axis=(0,1))
		arr = arr / np.std(arr, axis=(0,1),  ddof=1)

	x, y, n_bands = arr.shape
	plt.figure()
	plt.suptitle(filename)
	for b in range(n_bands):
		ax = plt.subplot(4, 3, b+1)
		ax.set_title(bands[b])
		data = arr[:,:,b]
		data = stretcher(data, clip=False)
		vmin, vmax = scaler.get_limits(data)
		print(vmin,vmax)
		ax.imshow(data, interpolation='nearest', cmap=plt.cm.gray_r, vmin=vmin, vmax=vmax) #norm=colors.PowerNorm(0.1))
		ax.axis('off')
	
	if plot_composites:
		im_lupton = make_lupton_rgb(arr[:,:,9], arr[:,:,7], arr[:,:,5], stretch=0.5) #750 rgu 975 gri
		ax = plt.subplot(4,3,b+2)
		ax.set_title('our gri composite')
		ax.axis('off')
		ax.imshow(im_lupton, interpolation='nearest', cmap=plt.cm.gray_r, norm=colors.PowerNorm(0.1))

		im_lupton = make_lupton_rgb(arr[:,:,7], arr[:,:,5], arr[:,:,0], stretch=0.5) #750 rgu
		ax = plt.subplot(4,4,b+3)
		ax.set_title('our rgu composite')
		ax.axis('off')
		ax.imshow(im_lupton, interpolation='nearest', cmap=plt.cm.gray_r, norm=colors.PowerNorm(0.1))
	
	plt.show()


def plot_rgb(filename):
	arr = np.load(filename)
	obj_id = filename.split('/')[-1].replace('.npy','')
	field = obj_id.split('.')[1]
	print(obj_id)
	im = cv2.imread('../raw-data/train_images/{}.trilogy.png'.format(field))
	cat = pd.read_csv('sloan_splus_matches.csv')
	obj = cat[cat['id'] == obj_id].iloc[0]
	x = obj['x'] #1341
	y = 11000 - obj['y'] #11000-3955
	d = 10
	print(x,y)
	obj = im[y-d:y+d,x-d:x+d]
	ax1 = plt.subplot(121)
	ax1.axis('off')
	ax1.imshow(obj)
	im_lupton = make_lupton_rgb(arr[:,:,7], arr[:,:,5], arr[:,:,0], stretch=1) #rgu
	ax2 = plt.subplot(122)
	ax2.imshow(im_lupton, interpolation='nearest', cmap=plt.cm.gray_r)
	ax2.axis('off')
	plt.show()


def magnitude_hist(filename):
	cat = pd.read_csv(filename)
	classes = cat['class'].unique()
	print(classes)
	for c in classes:
		cat.loc[cat['class']==c, 'r'].hist(bins=100, alpha=0.7)
	plt.legend(classes)
	plt.xlabel('MAGNITUDE')
	plt.ylabel('# OF SAMPLES')
	plt.show()


def update_legend_marker(handle, orig):
    handle.update_from(orig)
    handle.set_sizes([64])
    handle.set_alpha(1)


def plot_2d_embedding(X, y, filepath, format_='png', clear=True, labels=['galaxy', 'star', 'qso', 'unknown']):
	if clear:
		plt.clf()
	if y is not None:
		colors = ['m', 'b', 'y', 'k']
		for c in [3,0,1,2]: #np.unique(y):
			X_c = X[y==c]
			plt.scatter(X_c[:,0], X_c[:,1], c=colors[c], s=1, marker='.', alpha=0.15, label=labels[c])
		plt.legend(handler_map={PathCollection : HandlerPathCollection(update_func=update_legend_marker)})
	else:
		plt.scatter(X[:,0], X[:,1], c='k', s=0.5, marker='.', alpha=0.1) #c=k => black
	plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
	fig = plt.gcf()
	fig.set_size_inches(18.5, 10.5)
	plt.savefig(filepath+'.'+format_, dpi=100, format=format_)


# files = glob("../raw-data/coadded/*", recursive=True)
# im_path = files[0]
# print(im_path)
# print_single_band(im_path)

# field = 'STRIPE82-0003'
# im = cv2.imread('../raw-data/train_images/{}.trilogy.png'.format(field))
# plt.figure()
# plt.imshow(im)
# plt.show()

# plot_bands(filename)

# f = '../raw-data/coadded/STRIPE82-0003_{}_swp.fits.fz'
# plot_field_rgb(f)
# magnitude_hist('csv/matched_cat_dr1.csv')

cat_labeled = 'csv/dr1_matched.csv'
# cat_unlabeled = 'csv/diff_cat_dr1.csv' #'csv/matched_cat_dr1.csv'
_, y_true, _ = get_sets(cat_labeled)

# tsne
labeled = 'npy/dr1_features_tsne.npy'
unlabeled = 'npy/dr1_diff_features_tsne.npy'
X_labeled = np.load(labeled)
X_unlabeled = np.load(unlabeled)
idx = np.random.choice(X_unlabeled.shape[0], 200000, replace=False)
#X_unlabeled = X_unlabeled[idx]
X = np.vstack((X_labeled, X_unlabeled))
length = X.shape[0]
X = np.load('npy/dr1_full_features_tsne.npy')
#X = X[:length]
y = np.zeros((length,), dtype=np.int8)
# print(X.shape, y.shape)
y[:] = 3
y[:y_true.shape[0]] = y_true
plot_2d_embedding(X, y, 'tsne0821b')
# plot_2d_embedding(X_unlabeled, None, 'tsne0727')
# plot_2d_embedding(X_labeled, y_true, 'tsne0727', clear=False)

# umap
labeled = 'npy/dr1_features_umap.npy'
unlabeled = 'npy/dr1_diff_features_umap.npy'
X_labeled = np.load(labeled)
X_unlabeled = np.load(unlabeled)
print(X_unlabeled.shape)
idx = np.random.choice(X_unlabeled.shape[0], 200000, replace=False)
#X_unlabeled = X_unlabeled[idx]
print(X_unlabeled.shape)
X = np.vstack((X_labeled, X_unlabeled))
length = X.shape[0]
y = np.zeros((length,), dtype=np.int8)
y[:] = 3
y[:y_true.shape[0]] = y_true

X = np.load('npy/dr1_full_features_umap.npy')
#X = X[:length]
plot_2d_embedding(X, y, 'umap0821b')
# plot_2d_embedding(X_unlabeled, None, 'umap0727')
# plot_2d_embedding(X_labeled, y_true, 'umap0727', clear=False)

# print('saved embedding figures')
