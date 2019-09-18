from astropy.io import fits
from astropy.visualization import AsinhStretch
from cv2 import imread, imwrite, resize, INTER_CUBIC
from glob import glob
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import pandas as pd
import os
from shutil import copyfile
from time import time
from tqdm import tqdm


'''
dr0 LABELED
    fwhm >= 19   --> 146
    r <= 17      --> 8115
    r <= 18      --> 18455
    r <= 19      --> 32273
    r <= 20      --> 47911
    total        --> 61619

    r <= 20
    GALAXY    0.514809
    STAR      0.440191
    QSO       0.045000

'''


asinh_transform = AsinhStretch()

n_cores = multiprocessing.cpu_count()


def sample_move(df, src_base, dst_base, n_samples=50):
    idx = np.random.choice(df.shape[0], n_samples, replace=False)
    files = df.loc[idx, 'id'].values
    classes = df.loc[idx,'class'].values
    for ix, f in enumerate(files):
        src = src_base+f.split('/')[-1].split('.')[0]+'/'+f+'.png'
        dst = dst_base+f+'.'+classes[ix]+'.png'
        print(src, dst)
        copyfile(src, dst)


def recast_inner(file):
    arr = np.load(file)
    arr = np.float32(arr)
    np.save(file, arr)


def recast(folder_pattern):
    files = glob(folder_pattern)
    Parallel(n_jobs=8, prefer='threads')(delayed(recast_inner)(f) for f in tqdm(files))


def get_ndarray(filepath):
    fits_im = fits.open(filepath)
    return fits_im[1].data


def crop_objects_in_rgb(df, input_folder, save_folder, size=32, fwhm_radius=1.5):
    d = size//2
    print('df (original)', df.shape)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    df['X'] = df.x
    df['Y'] = 11000 - df.y
    df['field'] = df.id.apply(lambda s: s.split('.')[0])

    # ignore objects that have already been cropped
    imgfiles = glob(save_folder+'*.png', recursive=True)
    imgfiles = [i.split('/')[-1][:-4] for i in imgfiles]
    df = df[~df.id.isin(imgfiles)]
    print('df after ignoring existing crops', df.shape)
    
    lst_field = ''
    for ix, r in df.iterrows():
        field = r['field']
        if field != lst_field:
            imgfile = input_folder+'{}_trilogy.png'.format(field)
            print('cropping objects in', imgfile)
            fullimg = imread(imgfile)
            if not os.path.exists(save_folder+field):
                os.makedirs(save_folder+field)
        lst_field = field
        # d = np.ceil(np.maximum(radius, fwhm_radius*r['fwhm'])).astype(np.int)
        img = fullimg[r['Y']-d:r['Y']+d, r['X']-d:r['X']+d]
        if img.shape[0] != size or img.shape[1] != size:
            img = resize(img, dsize=(size, size), interpolation=INTER_CUBIC)
            print('resized', r['id'])
        # print('saving {}'.format(r['id']))
        imwrite('{}{}/{}.png'.format(save_folder, r['field'], r['id']), img)


def crop_object_in_field(ix, arr, objects_df, save_folder, asinh=True, size=32, radius=16, fwhm_radius=1.5):
    '''
    crops object in a given field
    receives:
        * ix            (int) index of object to be cropped in objects_df
        * arr           (ndarray) 12-band full field image
        * objects_df    (pandas DataFrame) full objects table
        * save_folder   (str) path to folder where crops will be saved

    for an object with fwhm =~ 2, 32x32 px is a good fit
    size=3*fwhm is good for larger objects (inspected visually) but yields too many resizes
    '''
    # d = np.ceil(np.maximum(delta, 0.5*o['fwhm'])).astype(np.int)
    row = objects_df.loc[ix]
    d = np.ceil(np.maximum(radius, fwhm_radius*row['fwhm'])).astype(np.int)
    x0 = np.maximum(0, int(row['x']) - d)
    x1 = np.minimum(10999, int(row['x']) + d)
    y0 = np.maximum(0, int(row['y']) - d)
    y1 = np.minimum(10999, int(row['y']) + d)
    im = arr[y0:y1, x0:x1, :]
    if im.shape[0] != size or im.shape[1] != size:
        im = resize(im, dsize=(size, size), interpolation=INTER_CUBIC)
        print('{} resized'.format(row['id']))
    if asinh:
        im = asinh_transform(im, clip=False)
    np.save('{}/{}.npy'.format(save_folder, row['id']), im)

    return 0


def get_bands_order():
    '''
    maps desired depthwise position to alphabetical index
    i.e., map
    0   F378
    1   F395
    2   F410
    3   F430
    4   F515
    5   F660
    6   F861 
    7   G
    8   I
    9   R
    10  U
    11  Z

    to U F378 F395 F410 F430 G F515 R F660 I F861 Z
    '''
    return [10, 0, 1, 2, 3, 7, 4, 9, 5, 8, 6, 11]



def sweep_fields(fields_path, catalog_path, crops_folder):
    '''
    sweeps field images cropping and saving objects in fields
    receives:
        * fields_path   (str) path pattern to get fits.fz field images
        * catalog_path  (str) catalog where x,y coordinates for objects are stored
        * crops_folder  (str) folder where image crops will be saved
    '''

    files = glob(fields_path, recursive=True)
    files.sort()

    print('reading catalog')
    catalog = pd.read_csv(catalog_path)

    # filter
    catalog = catalog[~catalog['class'].isna()]
    print(catalog.head())
    print('catalog shape', catalog.shape)

    catalog['field_name'] = catalog['id'].apply(lambda s: s.split('.')[0])

    # ignore objects that have already been cropped
    imgfiles = glob(crops_folder+'*/*.npy')
    imgfiles = [i.split('/')[-1][:-4] for i in imgfiles]
    print('catalog', catalog.shape)
    catalog = catalog[~catalog.id.isin(imgfiles)]
    print('catalog after ignoring existing crops', catalog.shape)
    fields = np.unique(catalog.field_name.values)
    files_orig = files
    files = []
    for field in fields:
        files_tmp = [f for f in files_orig if field in f]
        files = files +files_tmp
    del files_orig

    if len(files)==0:
        print('all objects already have crops')
        exit()

    bands_order = get_bands_order()
    n_channels = len(bands_order)

    data = get_ndarray(files[0])
    s0, s1 = data.shape
    arr = np.zeros((s0, s1, n_channels), dtype=np.float32)
    print('field array shape ', arr.shape)
    arr[:,:,bands_order[0]] = np.copy(data)
    prev = files[0].split('/')[-1].split('_')[0]

    start = time()
    i=1
    lst_ix = len(files[1:])-1
    for ix, f in enumerate(files[1:]):
        field_name = f.split('/')[-1].split('_')[0]
        if prev != field_name or ix==lst_ix:
            print('{} min. cropping objects in {}'.format(int((time()-start)/60), prev))
            objects_df = catalog[catalog.field_name==prev].reset_index()
            Parallel(n_jobs=8, prefer='threads')(delayed(
                crop_object_in_field)(ix, arr, objects_df, crops_folder+prev, True) for ix in range(objects_df.index.max()+1))
            arr = np.zeros((s0, s1, n_channels))
            prev = field_name
            i=0
        data = get_ndarray(f)
        arr[:,:,bands_order[i]] = np.copy(data)
        i+=1


def get_min_max(filefolder, n_channels=12):
    '''
        receives:
            * filefolder    (str) folder pattern wherein ndarray images are
            * n_channels    (int) number of channels in images
        returns:
            a tuple (minima, maxima), each an array of length=n_channels 
                    containing minima and maxima per band across all images 
    '''
    start = time()
    files = glob(filefolder)
    minima, maxima = np.zeros(n_channels), np.zeros(n_channels)
    min_files, max_files = ['']*n_channels, ['']*n_channels
    n_files = len(files)
    print('nr of files', n_files)
    for file in files:
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

    print('minutes taken:', int((time()-start)/60))
    print('minima', minima)
    print(min_files)
    print('maxima', maxima)
    print(max_files)

    return np.floor(minima), np.ceil(maxima)


def get_mean_var(filefolder, n_channels=12):
    '''
        receives:
            * filefolder    (str) folder pattern wherein ndarray images are
            * n_channels    (int) number of channels in images
        returns:
            a tuple (mean, var), each an array of length=n_channels 
                    containing mean and variance per band across all images
        reference:
            https://www.researchgate.net/post/How_to_combine_standard_deviations_for_three_groups
    '''
    start = time()
    files = glob(filefolder)
    mean, var = np.zeros(n_channels), np.zeros(n_channels)
    n_files = len(files)
    print('nr of files', n_files)
    for file in files:
        im = np.load(file)
        mean = mean + np.mean(im, axis=(0,1))
        var =  var + np.std(im, axis=(0,1))

    mean = mean/n_files
    var = var/n_files

    print('minutes taken:', int((time()-start)/60))
    print('means', mean)
    print('variances', var)

    return mean, var


def normalize_images(input_folder, output_folder, bounds_lower, bounds_upper):
    '''
    saves ndarray images resized to (32,32,n_channels) and normalized to values in [0,1]
    receives:
        * input_folder      (str) folder path wherein are (x,x,n_channels) ndarray images with varying shapes and value ranges
        * output_folder     (str) folder wherein normalized images will be saved
        * bounds_lower      (ndarray) (n_channels,) array that gives lower bounds for normalization
        * bounds_upper      (ndarray) (n_channels,) array that gives upper bounds for normalization 
    '''
    files = glob(input_folder)
    print('nr of files', len(files))

    interval = bounds_upper - bounds_lower
    interval = interval[None,None,:]
    lower = bounds_lower[None,None,:]

    start = time()
    for file in files:
        im = np.load(file)
        im = im - lower
        im = im / interval
        if im.min() < 0 or im.max() > 1:
            print('{} out of [0,1] range'.format(file.split('/')[-1]))
        np.save('{}{}'.format(output_folder, file.split('/')[-1]), im, allow_pickle=False)
    print('minutes taken:', int((time()-start)/60))


def z_norm_images(input_folder, output_folder):
    '''
    saves ndarray images resized to (32,32,n_channels) and normalized by their z-score: x-mean/std
    receives:e
        * input_folder      (str) folder path wherein are (x,x,n_channels) ndarray images with varying shapes and value ranges
        * output_folder     (str) folder wherein normalized images will be saved
    '''
    files = glob(input_folder)
    print('nr of files', len(files))

    start = time()
    for file in files:
        im = np.load(file)
        im = im - np.mean(im, axis=(0,1))
        im = im / np.std(im, axis=(0,1))
        if im.shape[0] > 32:
            im = resize(im, dsize=(32, 32), interpolation=INTER_CUBIC)
            # print('{} resized'.format(file.split('/')[-1]))
        np.save('{}{}'.format(output_folder, file.split('/')[-1]), im, allow_pickle=False)
    print('minutes taken:', int((time()-start)/60))


if __name__=='__main__':
    data_dir = os.environ['DATA_PATH']
    # recast(data_dir+'/crops_asinh/*/*.npy')

    df = pd.read_csv('csv/dr1_classes_split.csv')
    df = df[(df.ndet==12)&(df.photoflag==0)]
    crop_objects_in_rgb(df, data_dir+'/dr1/color_images/', data_dir+'/crops_rgb32/')
    exit()

    sweep_fields(
        fields_path=data_dir+'/dr1/coadded/*/*.fz',
        catalog_path='csv/dr1_classes_split.csv',
        crops_folder=data_dir+'/crops_asinh/'
        )