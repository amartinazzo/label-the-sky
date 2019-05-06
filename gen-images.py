from astropy.io import fits
from glob import glob
import numpy as np
import pandas as pd
import os
from time import time


delta = 16


def move_broadbands(images_folder='../raw-data/early-dr/coadded/*'):
	files = glob(images_folder, recursive=False)
	bands = ['_U_', '_G_', '_R_', '_I_', '_Z_']
	for file in files:
		for b in bands:
			if b in file:
				split = file.split('/')
				print('moving: '+split[0]+'/broadbands/'+split[1])
				os.rename(file, split[0]+'/broadbands/'+split[1])


def get_ndarray(filepath):
	fits_im = fits.open(filepath)
	return fits_im[1].data


def crop_objects_in_field(arr, catalog, save_folder, field_name=None):
	objects = catalog[catalog.field_name==field_name]
	for ix, o in objects.iterrows():
		d = np.ceil(np.maximum(delta, o['fwhm']/2)).astype(np.int)
		x0 = int(o['x']) - d
		x1 = int(o['x']) + d
		y0 = int(o['y']) - d
		y1 = int(o['y']) + d
		cropped = arr[y0:y1, x0:x1, :]
		subfolder = '32x32' if d==32 else 'larger'
		np.save('{}{}/{}.npy'.format(save_folder, subfolder, o['id']), cropped, allow_pickle=False)


def get_bands_order():
	'''
	maps desired depthwise position to alphabetical index
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
	return [10, 0, 1, 2, 3, 7, 4, 9, 5, 8, 6, 11]



def sweep_fields(fields_path, catalog_path, crops_folder):
	'''
	sweeps field images cropping and saving objects in fields
	receives:
		* fields_path	path pattern to get fits.fz field images
		* catalog_path	catalog where x,y coordinates for objects are stored
		* crops_folder	folder where image crops will be saved
	'''

	files = glob(fields_path, recursive=True)
	files.sort()

	print('reading catalog')
	catalog = pd.read_csv(catalog_path)
	print(catalog.head())
	print('catalog shape', catalog.shape)
	catalog['field_name'] = catalog.id.apply(lambda s: s.split('.')[1])

	# print('filtering objects with fwhm<=', max_fwhm)
	# catalog = catalog[catalog.fwhm<=max_fwhm]

	bands_order = get_bands_order()
	n_channels = len(bands_order)

	data = get_ndarray(files[0])
	s0, s1 = data.shape
	arr = np.zeros((s0, s1, n_channels))
	print('field array shape ', arr.shape)
	arr[:,:,bands_order[0]] = np.copy(data)
	prev = files[0].split('/')[-1].split('_')[0]

	start = time()
	i=1
	for ix, f in enumerate(files[1:]):
		field_name = f.split('/')[-1].split('_')[0]
		if prev != field_name or ix==len(files[1:]):
			print('{} min. cropping objects in {}'.format(int((time()-start)/60), prev))
			crop_objects_in_field(arr, catalog, crops_folder, prev)
			arr = np.zeros((s0, s1, n_channels))
			prev = field_name
			i=0
		data = get_ndarray(f)
		arr[:,:,bands_order[i]] = np.copy(data)
		i+=1



if __name__=='__main__':
	sweep_fields(
		fields_path='../raw-data/dr1/coadded/*/*.fz',
		catalog_path='csv/matched_cat_dr1_filtered.csv',
		crops_folder='../raw-data/crops/'
		)
