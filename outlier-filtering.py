from glob import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np
from time import time


filepattern = '../raw-data/crops/*.npy'
file = '../raw-data/crops/SPLUS.STRIPE82-0058.01798.npy'


def plot_3d(img):
    x, y = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(x, y, img, rstride=1, cstride=1, cmap=plt.cm.gray, linewidth=0)
    plt.show()


# from http://bugra.github.io/work/notes/2014-03-31/outlier-detection-in-time-series-signals-fft-median-filtering/
# TODO: what would be a good threshold?
def get_median_filtered(signal, threshold=5):
    signal = signal.copy()
    difference = np.abs(signal - np.median(signal))
    median_difference = np.median(difference)
    if median_difference == 0:
        s = 0
    else:
        s = difference / float(median_difference)
    mask = s > threshold
    signal[mask] = np.median(signal)
    return signal


def get_min_max(filefolder, n_channels=12):
	start = time()
	files = glob(filefolder)
	minima, maxima = np.zeros(n_channels), np.zeros(n_channels)
	min_files, max_files = ['']*n_channels, ['']*n_channels
	n_files = len(files)
	print('nr of files', n_files)
	for ix, file in enumerate(files):
		im = np.load(file)
		min_tmp = np.min(im, axis=(0,1))
		max_tmp =  np.max(im, axis=(0,1))

		msk = np.less(min_tmp, minima)
		if msk.any():
			minima[msk] = min_tmp[msk]
			min_files = [file if msk[i] else min_files[i] for i in range(n_channels)]

		msk = np.greater(max_tmp, maxima)
		if msk.any():
			maxima[msk] = max_tmp[msk]
			max_files = [file if msk[i] else max_files[i] for i in range(n_channels)]
		# display progress in 10% steps
		# r = np.round(ix/n_files,2)
		# if r>0 and r % 0.1 == 0:
		# 	print('processed {} after {} min'.format(r, int((time()-start)/60)))
	print('minutes taken:', int((time()-start)/60))

	print('minima', minima)
	print(min_files)
	print('maxima', maxima)
	print(max_files)


get_min_max(filepattern)

# print(file)
# im = np.load(file) #channels-last; idx 7 is R
# idx_max = np.unravel_index(np.argmax(im, axis=None), im.shape)

# # get min and max along each channel
# print('min:', np.min(im, axis=(0,1)))
# print('max:', np.max(im, axis=(0,1)))
# print('argmax:', idx_max)

# # returns one band (the one that has the maximum) filtered by the median
# im_filtered = get_median_filtered(im[:,:,idx_max[2]])
# # plot_3d(im[:,:,idx_max[2]])
# # plot_3d(im_filtered)
