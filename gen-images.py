from astropy.io import fits
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


delta = 10 # max fwhm in dataset is 19. side of image will be 2*delta


def move_broadbands():
	files = glob("../raw-data/coadded/*", recursive=False)
	bands = ['_U_', '_G_', '_R_', '_I_', '_Z_']
	for file in files:
		for b in bands:
			if b in file:
				split = file.split('/')
				print("moving: "+split[0]+"/broadbands/"+split[1])
				os.rename(file, split[0]+"/broadbands/"+split[1])


def get_ndarray(filepath):
	fits_im = fits.open(filepath)
	return fits_im[1].data


def crop_objects_in_field(arr, catalog, field_name=None):
	objects = catalog[catalog.field_name==field_name]
	for ix, o in objects.iterrows():
		x0 = int(o['x'] - delta)
		x1 = int(o['x'] + delta)
		y0 = int(o['y'] - delta)
		y1 = int(o['y'] + delta)
		cropped = arr[y0:y1, x0:x1, :]
		np.save("../raw-data/crops/{}.npy".format(o['id']), cropped, allow_pickle=False)


if __name__=="__main__":
	files = glob("../raw-data/coadded/*.fits.fz", recursive=True)
	files.sort()

	catalog = pd.read_csv("sloan_splus_matches.csv")
	print(catalog.head())
	print('catalog shape', catalog.shape)
	catalog['field_name'] = catalog.id.apply(lambda s: s.split(".")[1])
	#catalog['id'] = catalog['id'].apply(lambda s: s.replace('.griz',''))

	'''
	map desired depthwise position to alphabetical index
	i.e., map
	0	F378
	1	F395
	2	F410
	3	F430
	4	F515
	5	F660
	6	F861 
	7	G
	8	I
	9	R
	10	U
	11	Z

	to U F378 F395 F410 F430 G F515 R F660 I F861 Z
	'''
	bands_order = [10, 0, 1, 2, 3, 7, 4, 9, 5, 8, 6, 11]
	n_channels = len(bands_order)

	data = get_ndarray(files[0])
	s0, s1 = data.shape
	arr = np.zeros((s0, s1, n_channels))
	print('field array shape ', arr.shape)
	arr[:,:,bands_order[0]] = np.copy(data)
	prev = files[0].split('/')[3].split('_')[0]

	i=1
	for ix, f in enumerate(files[1:]):
		field_name = f.split('/')[3].split('_')[0]
		if prev != field_name or ix==len(files[1:]):
			print("cropping objects in {}".format(prev))
			crop_objects_in_field(arr, catalog, prev)
			arr = np.zeros((s0, s1, n_channels))
			prev = field_name
			i=0
		data = get_ndarray(f)
		arr[:,:,bands_order[i]] = np.copy(data)
		i+=1

