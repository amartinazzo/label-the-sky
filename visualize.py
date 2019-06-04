from astropy import visualization as vis
from astropy.io import fits
from astropy.visualization import make_lupton_rgb
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd

matplotlib.rcParams.update({'font.size': 6})

filename = '../raw-data/crops/normalized/SPLUS.STRIPE82-0042.10742.npy'


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
def plot_bands(arr, plot_trilogy=False, base_catalog='csv/sloan_splus_matches.csv'):
	arr = np.load(filename)
	x, y, n_bands = arr.shape
	plt.figure()
	plt.suptitle(filename)
	for b in range(n_bands):
		ax = plt.subplot(4, 4, b+1)
		ax.set_title(bands[b])
		data = arr[:,:,b]
		data = stretcher(data, clip=False)
		vmin, vmax = scaler.get_limits(data)
		ax.imshow(data, interpolation='nearest', cmap=plt.cm.gray_r, vmin=vmin, vmax=vmax) #norm=colors.PowerNorm(0.1))
		ax.axis('off')
	
	im_lupton = make_lupton_rgb(arr[:,:,9], arr[:,:,7], arr[:,:,5], stretch=0.5) #750 rgu 975 gri
	ax = plt.subplot(4,4,b+2)
	ax.set_title('our gri composite')
	ax.axis('off')
	ax.imshow(im_lupton, interpolation='nearest', cmap=plt.cm.gray_r, norm=colors.PowerNorm(0.1))

	im_lupton = make_lupton_rgb(arr[:,:,7], arr[:,:,5], arr[:,:,0], stretch=0.5) #750 rgu
	ax = plt.subplot(4,4,b+3)
	ax.set_title('our rgu composite')
	ax.axis('off')
	ax.imshow(im_lupton, interpolation='nearest', cmap=plt.cm.gray_r, norm=colors.PowerNorm(0.1))
	
	if plot_trilogy:
		obj_id = filename.split('/')[-1].replace('.npy','')
		field = obj_id.split('.')[1]
		im = cv2.imread('../raw-data/train_images/{}.trilogy.png'.format(field))
		cat = pd.read_csv(base_catalog)
		obj = cat[cat['id'] == obj_id].iloc[0]
		x = obj['x'] #1341
		y = 11000 - obj['y'] #11000-3955
		d = 10
		obj = im[y-d:y+d,x-d:x+d]
		ax = plt.subplot(4,4,b+4)
		ax.set_title('SPLUS composite')
		ax.axis('off')
		ax.imshow(obj)
	
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



# files = glob("../raw-data/coadded/*", recursive=True)
# im_path = files[0]
# print(im_path)
# print_single_band(im_path)

# field = 'STRIPE82-0003'
# im = cv2.imread('../raw-data/train_images/{}.trilogy.png'.format(field))
# plt.figure()
# plt.imshow(im)
# plt.show()

file = '../raw-data/crops/SPLUS.STRIPE82-0033.05627.npy'
plot_bands(filename)

# f = '../raw-data/coadded/STRIPE82-0003_{}_swp.fits.fz'
# plot_field_rgb(f)
# magnitude_hist('csv/matched_cat_dr1.csv')