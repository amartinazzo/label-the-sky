from astropy.io import fits
from glob import glob
import numpy as np
import pandas as pd
import os


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


def crop_objects_in_field(arr, catalog, field_name=None, save_field=False):
	if save_field:
		np.save("npy_fields/{}.npy".format(field), arr, allow_pickle=False)

	objects = catalog[catalog.field_name==field_name]
	objects = objects[['x', 'y', 'fwhm', 'id']]
	for ix, o in objects.iterrows():
		delta = o['fwhm']/2
		x0 = int(o['x'] - delta)
		x1 = int(o['x'] + delta)
		y0 = int(o['y'] - delta)
		y1 = int(o['y'] + delta)
		cropped = arr[x0:x1, y0:y1]
		np.save("npy_objects/{}.npy".format(o.id), cropped, allow_pickle=False)


if __name__=="__main__":
	files = glob("../raw-data/coadded/*.fits.fz", recursive=True)
	files.sort()
	# print(files[:15])

	catalog = pd.read_csv("matched_cat.csv")
	catalog = catalog[catalog.matched]
	print('catalog shape ', catalog.shape)
	catalog['field_name'] = catalog.id.apply(lambda s: s.split(".")[1])

	'''
	map desired depthwise position to alphabetical index
	i.e., map 
	0	G
	1	I
	2	J0378
	3	J0395
	4	J0410
	5	J0430
	6	J0515
	7	J0660
	8	J0861
	9	R
	10	U
	11	Z

	to U J0378 J0395 J0395 J0410 J0430 G J0515 R J0660 I J0861 Z
	'''
	bands_order = [10, 2, 3, 4, 5, 0, 6, 9, 7, 1, 8, 11]
	n_channels = len(bands_order)

	data = get_ndarray(files[0])
	s0, s1 = data.shape
	arr = np.zeros((n_channels, s0, s1), dtype=np.int32)
	print('field array shape ', arr.shape)
	arr[bands_order[0],:,:] = data
	prev = files[0].split('/')[3].split('_')[0]

	i=1
	for ix, f in enumerate(files[1:]):
		field_name = f.split('/')[3].split('_')[0]
		if prev != field_name or ix==len(files[1:]):
			print("cropping objects in {}".format(field_name))
			crop_objects_in_field(arr, catalog, field_name)
			arr = np.zeros((n_channels, s0, s1), dtype=np.int32)
			prev = field_name
			i=0
		data = get_ndarray(f)
		arr[bands_order[i],:,:] = data
		i+=1

