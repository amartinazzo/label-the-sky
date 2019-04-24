# from astropy import units as u
# from glob import glob
import pandas as pd
import numpy as np
import time


# nohup python3 -u preprocess-spectra.py > spectra-preprocessing.log&


def concat_frames(files_glob_str):
	files = glob(files_glob_str, recursive=False)
	print('nr of files in {}'.format(files_glob_str), len(files))
	li = []
	for f in files:
	    df = pd.read_csv(f, index_col=None, header=0)
	    li.append(df)
	frame = pd.concat(li, axis=0, ignore_index=True)
	print('frame shape', frame.shape)


start = time.perf_counter()
print('loading loglamb matrix')
lambs = pd.read_csv('spectra/loglamb.dat',sep=' ', header=None)
print('time taken (min)', (time.perf_counter()-start)/60)

lambs = lambs.values
lambs_unique = np.unique(lambs)
lambs_unique = lambs_unique[lambs_unique>0]

base_series = pd.Series(index=lambs_unique)

object_idx = pd.read_csv('spectra/object_idx.csv', header=None)
object_idx = object_idx[1].values

print('loading flux matrix')
fluxes = pd.read_csv('spectra/fluxes.dat', sep=' ', header=None)
fluxes = fluxes.values
print('time taken (min)', (time.perf_counter()-start)/60)

n, total_length = fluxes.shape

for i in range(n):
	print('processing', object_idx[i])
	if lambs[i,0] == 0:
		print('skip: zero array.')
		continue
	nonzero_idx = np.nonzero(lambs[i])[0]
	if len(nonzero_idx)/total_length < 0.3:
		print('skip: sparse series.')
		continue
	series = pd.Series(data=fluxes[i, nonzero_idx], index=lambs[i, nonzero_idx])
	series = series.combine_first(base_series)
	series.interpolate(method='spline', order=1, limit_direction='both', inplace=True)
	np.savetxt('spectra/flux-processed/{}.dat'.format(object_idx[i]), series.values)

print('time taken (min)', (time.perf_counter()-start)/60)