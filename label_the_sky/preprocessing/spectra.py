from glob import glob
import pandas as pd
import numpy as np
import os
import time


sparse_thres = 0.3
lamb_lower = 3750
lamb_upper = 9250
lambs_interval = np.arange(lamb_lower, lamb_upper)


def concat_frames(files_glob_str):
    files = glob(files_glob_str, recursive=False)
    print('nr of files in {}'.format(files_glob_str), len(files))
    li = []
    for f in files:
        df = pd.read_csv(f, index_col=None, header=0)
        li.append(df)
    frame = pd.concat(li, axis=0, ignore_index=True)
    print('frame shape', frame.shape)


def get_unique_lambs(unique_loglambs_filepath, loglambs_filepath):
    if os.path.exists(unique_loglambs_filepath):
        return np.load(unique_loglambs_filepath)
    lambs = pd.read_csv(loglambs_filepath, sep=' ', header=None)
    lambs = lambs.values
    lambs = np.unique(lambs)
    lambs = lambs[lambs>0]
    np.save(unique_loglambs_filepath, lambs)
    return lambs


def to_angstrom(loglamb):
    lamb = np.round(10**loglamb).astype(int)
    lamb = lamb[(lamb>=lamb_lower) & (lamb<=lamb_upper)]
    lamb, unique_idx = np.unique(lamb, return_index=True)
    if lamb[0] == 1:
        lamb = lamb[1:]
        unique_idx = unique_idx[1:]
    return lamb, unique_idx


def gen_series(flux_arr, lamb_arr, base_series):
    series = pd.Series(data=flux_arr, index=lamb_arr)
    series = series.combine_first(base_series)
    series.interpolate(method='spline', order=1, limit_direction='both', inplace=True)
    return series


def preprocess_matrices(loglambs_filepath, fluxes_filepath, objects_filepath):
    start = time.perf_counter()
    print('loading loglamb matrix')
    lambs = pd.read_csv(loglambs_filepath,sep=' ', header=None)
    lambs = lambs.values
    print('time taken (min)', (time.perf_counter()-start)/60)

    base_series = pd.Series(index=lambs_interval)

    object_idx = pd.read_csv(objects_filepath, header=None)
    object_idx = object_idx[1].values

    print('loading flux matrix')
    fluxes = pd.read_csv(fluxes_filepath, sep=' ', header=None)
    fluxes = fluxes.values
    print('time taken (min)', (time.perf_counter()-start)/60)

    n, total_length = fluxes.shape

    for i in range(n):
        print('processing', object_idx[i])
        if lambs[i,0] == 0:
            print('skip: zero array.')
            continue
        lamb, unique_idx = to_angstrom(lambs[i])
        if len(unique_idx)/total_length < sparse_thres:
            print('skip: sparse series.')
            continue
        series = gen_series(fluxes[i, unique_idx], lamb, base_series)
        np.savetxt('spectra/flux-processed/{}.txt'.format(object_idx[i]), series.values)

    print('time taken (min)', (time.perf_counter()-start)/60)


def preprocess_files(loglambs_folder, fluxes_folder):
    start = time.perf_counter()

    base_series = pd.Series(index=lambs_interval)

    total_length = len(lambs_interval)
    print('sequence length', total_length)

    flux_files = glob(fluxes_folder)
    loglamb_files = glob(loglambs_folder)
    flux_files.sort()
    loglamb_files.sort()

    print('nr of spectra to process', len(flux_files))

    for ix, file in enumerate(flux_files):
        object_id = file.split('/')[-1][:-4]
        print('processing', file)
        flux = np.loadtxt(file)
        lamb = np.loadtxt(loglamb_files[ix])
        lamb, unique_idx = to_angstrom(lamb)

        if len(lamb)/total_length < sparse_thres:
            print('skip: sparse sequence.')
            continue

        series = gen_series(flux[unique_idx], lamb, base_series)
        assert len(series)==len(base_series)
        np.savetxt('spectra/flux-processed/{}.txt'.format(object_id), series.values)

        if ix % 5000 == 0:
            print('time taken (min)', (time.perf_counter()-start)/60)           

    print('time taken (min)', (time.perf_counter()-start)/60)


def get_min_max(filefolder):
    '''
        receives:
            * filefolder    (str) folder pattern wherein txt spectra are
        returns:
            a tuple (minimum, maximum) 
    '''
    start = time.perf_counter()
    files = glob(filefolder)
    minimum, maximum = 1000, 0
    min_file, max_file = '', ''
    n_files = len(files)
    print('nr of files', n_files)
    for file in files:
        spectra = np.loadtxt(file)
        min_tmp = np.min(spectra)
        max_tmp =  np.max(spectra)

        if min_tmp < minimum:
            minimum = min_tmp
            min_file = file

        if max_tmp > maximum:
            maximum = max_tmp
            max_file = file

    print('minutes taken:', int((time.perf_counter()-start)/60))
    print('minimum : {} at {}'.format(minimum, min_file))
    print('maximum : {} at {}'.format(maximum, max_file))

    return np.floor(minimum), np.ceil(maximum)


def normalize(input_folder, output_folder, bound_lower, bound_upper):
    '''
    saves spectra normalized to values in [0,1]
    receives:
        * input_folder      (str) folder path wherein are 1-d arrays in txt files with varying ranges
        * output_folder     (str) folder wherein normalized txt will be saved
        * bounds_lower      (float) lower bound for normalization
        * bounds_upper      (float) upper bound for normalization 
    '''
    files = glob(input_folder)
    print('nr of files', len(files))

    interval = bound_upper - bound_lower
    lower = bound_lower

    start = time()
    for file in files:
        spectra = np.loadtxt(file)
        spectra = spectra - lower
        spectra = spectra / interval
        if spectra.min() < 0 or spectra.max() > 1:
            print('{} out of [0,1] range'.format(file.split('/')[-1]))
        np.savetxt('{}{}'.format(output_folder, file.split('/')[-1]), spectra)
    print('minutes taken:', int((time()-start)/60))
