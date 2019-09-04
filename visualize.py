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
import os
from utils import get_sets


matplotlib.rcParams.update({'font.size': 6})
bands = ['U', 'F378', 'F395', 'F410', 'F430', 'G', 'F515', 'R', 'F660', 'I', 'F861', 'Z']

stretcher = vis.AsinhStretch()
# scaler = vis.ZScaleInterval()


def plot_single_band(im_path):
	fits_im = fits.open(im_path)
	data = fits_im[1].data #ndarray
	vmin = data.min()
	vmax = data.max()
	print(vmin,vmax)
	# std = (data - vmin) / (vmax-vmin)
	# data = std * (vmax - vmin) + vmin
	# data = data[650:750, 450:550]
	# data = data[3500:4500, 7000:8000]
	data = data[8300:8600,9100:9400]
	data = stretcher(data, clip=False)
	print(data.min(), data.max())
	plt.imshow(data, cmap='gray')
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
def plot_bands(filename, cols=6, rows=2, plot_composites=False, znorm=False):
	arr = np.load(filename)
	if znorm:
		arr = arr - np.mean(arr, axis=(0,1))
		arr = arr / np.std(arr, axis=(0,1),  ddof=1)

	x, y, n_bands = arr.shape
	plt.figure(figsize=(400/100, 50/100), dpi=100) #size: px/dpi
	plt.figure()
	# plt.title(filename.split('/')[-1])
	for b in range(n_bands):
		ax = plt.subplot(rows, cols, b+1)
		ax.set_title(bands[b])
		data = arr[:,:,b]
		# data = stretcher(data, clip=False)
		# vmin, vmax = scaler.get_limits(data)
		# print(vmin,vmax)
		ax.imshow(data, cmap=plt.cm.gray_r)#, norm=colors.PowerNorm(1))
		ax.axis('off')
		ax.axis('scaled')
	
	if plot_composites:
		# im = make_lupton_rgb(arr[:,:,9], arr[:,:,7], arr[:,:,5]) #750 rgu 975 gri
		im = np.dstack((arr[:,:,9], arr[:,:,7], arr[:,:,5]))
		ax = plt.subplot(rows, cols,b+2)
		ax.set_title('GRI composite')
		ax.axis('off')
		ax.imshow(im)

		# im = make_lupton_rgb(arr[:,:,7], arr[:,:,5], arr[:,:,0]) #750 rgu
		im = np.dstack((arr[:,:,7], arr[:,:,5], arr[:,:,0]))
		ax = plt.subplot(rows, cols,b+3)
		ax.set_title('RGU composite')
		ax.axis('off')
		ax.imshow(im)
	
	plt.tight_layout()
	plt.savefig(filename.split('/')[-1][:-4]+'_12bands.png', bbox_inches='tight')


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


'''
colors
b: blue
g: green
r: red
c: cyan
m: magenta
y: yellow
'''

def plot_2d_embedding(X, df, filepath, format_='png', clear=True):
	if clear:
		plt.clf()
	if df is not None:
		colors = ['b', 'c', 'r', 'm', 'y', 'g']
		groups = list(np.unique(df.class_split))
		groups.sort()
		for c, group in enumerate(groups): #np.unique(y):
			X_c = X[df['class_split']==group]
			# print(group, X_c.shape)
			plt.scatter(X_c[:,0], X_c[:,1], c=colors[c], s=3, marker='.', alpha=0.1, label=groups[c])
		plt.legend(handler_map={PathCollection : HandlerPathCollection(update_func=update_legend_marker)})
	else:
		plt.scatter(X[:,0], X[:,1], c='k', s=0.5, marker='.', alpha=0.1) #c=k => black
	plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
	fig = plt.gcf()
	plt.tight_layout()
	plt.savefig(f'{filepath}.{format_}', dpi=100, format=format_)


if __name__ == '__main__':
	csv_dataset = 'csv/dr1_classes_split.csv' #dr1_classes_mag1418_split_ndet.csv'
	df = pd.read_csv(csv_dataset)
	df = df[(df.split!='test')] #& (df.n_det==12) & (~df['class'].isna())]
	df['class_split'] = df['class']+' '+df.split

	for m in ['classification_12bands', 'classification_3bands', 'regression_12bands', 'regression_3bands']: #['classification', 'regression']
		print('plotting', m)
		name = f'umap_avgpool_{m}'
		# name = f'umap_avgpool_depth11_card4_eph500_{m}_mag1418'
		X = np.load(f'npy/{name}.npy')
		plot_2d_embedding(X, df, f'vis/{name}')