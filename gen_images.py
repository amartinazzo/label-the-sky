from astropy.io import fits
from glob import glob
import numpy as np
import os


def move_broadbands():
	files = glob("coadded/*", recursive=False)
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


if __name__=='__main__':
	files = glob("coadded/broadbands/*")
	files.sort()
	print(files)

	first=True
	bands_order = [1, 3, 2, 0, 4] # G I R U Z

	data = get_ndarray(files[0])
	s0, s1 = data.shape
	arr = np.array((5, s0, s1))
	arr[bands_order[0],:,:] = data
	prev = file.split('_')[0]

	i=1
	for file in files[1:]:
		field = file.split('_')[0]
		if prev != field:
			arr.save("npy_broadbands/{}.npy".format(field), allow_pickle=False)
			arr = np.array((5, s0, s1))
			i=0
		data = get_ndarray(file)
		arr[bands_order[i],:,:] = data
		i+=1

