from astropy.io import fits
from astropy.table import Table
from astropy.visualization import AsinhStretch
from cv2 import imread, imwrite, resize, INTER_CUBIC
from glob import glob
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import os
from time import time


asinh_transform = AsinhStretch()


def get_metadata(folder_pattern, save_file):
    files = glob(folder_pattern)
    files.sort()
    print('nr of files', len(files))
    dates = []
    airmasses = []
    for f in files:
        im = fits.open(f)
        dates.append(im[1].header['HIERARCH OAJ PRO REFDATEOBS'])
        airmasses.append(im[1].header['HIERARCH OAJ PRO REFAIRMASS'])
    df = pd.DataFrame({'file': files, 'date': dates, 'airmass': airmasses})
    df.to_csv(save_file, index=False)
    print('saved csv')
    df['file'] = df.file.apply(lambda s: s.split('/')[-1])
    df['field'] = df.file.apply(lambda s: s.split('_')[0])
    df['band'] = df.file.apply(lambda s: s.split('_')[1])
    df['year'] = df.date.apply(lambda s: s.split('-')[0])
    df.to_csv(save_file, index=False)
    print('saved csv again')


def crop_objects_in_rgb(catalog_path, input_folder, save_folder, size=32, fwhm_radius=1.5):
    d = size // 2
    df = pd.read_csv(catalog_path)
    print('df (original)', df.shape)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    df['X'] = df.x
    df['Y'] = 11000 - df.y
    df['field'] = df.id.apply(lambda s: s.split('.')[0])

    # ignore objects that have already been cropped
    imgfiles = glob(save_folder + '*/*.png')
    imgfiles = [i.split('/')[-1][:-4] for i in imgfiles]
    df = df[~df.id.isin(imgfiles)]
    print('df after ignoring existing crops', df.shape)

    df = df.sort_values(by='id')

    lst_field = ''
    for ix, r in df.iterrows():
        field = r['field']
        if field != lst_field:
            imgfile = input_folder + '{}_trilogy.png'.format(field)
            print('cropping objects in', imgfile)
            fullimg = imread(imgfile)
            if not os.path.exists(save_folder + field):
                os.makedirs(save_folder + field)
        lst_field = field
        # d = np.ceil(np.maximum(radius, fwhm_radius * r['fwhm'])).astype(np.int)
        img = fullimg[r['Y'] - d:r['Y'] + d, r['X'] - d:r['X'] + d]
        if img.shape[0] != size or img.shape[1] != size:
            img = resize(img, dsize=(size, size), interpolation=INTER_CUBIC)
            print('resized', r['id'])
        # print('saving {}'.format(r['id']))
        imwrite('{}{}/{}.png'.format(save_folder, r['field'], r['id']), img)


def crop_object_in_field(
        obj_ix, arr, objects_df, save_folder, asinh=True,
        size=32, radius=16, fwhm_radius=1.5):
    """
    crops object in a given field
    receives:
        * obj_ix        (int) index of object to be cropped in objects_df
        * arr           (ndarray) 12-band full field image
        * objects_df    (pandas DataFrame) full objects table
        * save_folder   (str) path to folder where crops will be saved

    for an object with fwhm =~ 2, 32x32 px is a good fit.
    size=3*fwhm is good for larger objects (inspected visually)
    but yields too many resizes.
    """
    row = objects_df.loc[obj_ix]
    d = np.ceil(np.maximum(radius, fwhm_radius * row['fwhm'])).astype(np.int)
    d = np.minimum(d, 75)  # 75 = 3 *(largest fwhm with photoflag==0) / 2
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
    """
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
    """
    return [10, 0, 1, 2, 3, 7, 4, 9, 5, 8, 6, 11]


def get_bands():
    return ['U', 'F378', 'F395', 'F410', 'F430', 'G',
            'F515', 'R', 'F660', 'I', 'F861', 'Z']


def get_zps(field):
    zpfile = os.path.join(
        os.environ['DATA_PATH'],
        'dr1/ZPfiles_Feb2019', '{}_ZP.cat'.format(field))
    zpdata = Table.read(zpfile, format="ascii")
    zps = dict([(t['FILTER'], t['ZP']) for t in zpdata])
    return zps


def make_calibration(data, zp):
    """
    applies corrections to given data (image) according to given zp (zero-point) value
    receives:
        * data (np array)  single band bidimensional image
        * zp   (float)     zero point value to be used for calibrating the image
    returns :
        * S    (np array)  bidimensional calibrated image
    """
    ps = 0.55  # pixel scale [arcsec / pixel]
    fnu = data * np.power(10, -0.4 * zp)  # spectral flux density
    # Surface brightness using fnu
    S = 1e5 * fnu / ps**2  # [1e5 erg / (s cm^2 Hz arcsec^2)]
    return S


def sweep_fields(fields_path, catalog_path, crops_folder, calibrate=True, asinh=False):
    """
    sweeps field images cropping and saving objects in fields
    receives:
        * fields_path   (str) path pattern to get fits.fz field images
        * catalog_path  (str) catalog where x,y coordinates for objects are stored
        * crops_folder  (str) folder where image crops will be saved
    """

    files = glob(fields_path, recursive=True)
    files.sort()

    print('reading catalog')
    df = pd.read_csv(catalog_path)

    # filter
    print('df shape', df.shape)

    df['field_name'] = df['id'].apply(lambda s: s.split('.')[0])

    # ignore objects that have already been cropped
    imgfiles = glob(crops_folder + '*/*.npy')
    imgfiles = [i.split('/')[-1][:-4] for i in imgfiles]
    print('df', df.shape)
    df = df[~df.id.isin(imgfiles)]
    print('df after ignoring existing crops', df.shape)
    fields = np.unique(df.field_name.values)
    files_orig = files
    files = []
    for field in fields:
        files_tmp = [f for f in files_orig if field in f]
        files = files + files_tmp
    del files_orig

    if len(files) == 0:
        print('all objects already have crops')
        return

    bands_order = get_bands_order()
    bands = get_bands()
    n_channels = len(bands_order)

    data = fits.getdata(files[0])
    s0, s1 = data.shape
    arr = np.zeros((s0, s1, n_channels), dtype=np.float32)
    print('field array shape ', arr.shape)

    prev = files[0].split('/')[-1].split('_')[0]
    if calibrate:
        zps = get_zps(prev)
        data = make_calibration(data, zps[bands[0]])
    arr[:, :, bands_order[0]] = np.copy(data)

    start = time()
    i = 1
    lst_ix = len(files[1:]) - 1
    for ix, f in enumerate(files[1:]):
        field_name = f.split('/')[-1].split('_')[0]
        if prev != field_name or ix == lst_ix:
            print('{} min. cropping objects in {}'.format(
                int((time() - start) / 60), prev))
            print('min: {}, max: {}'.format(arr.min(), arr.max()))
            objects_df = df[df.field_name == prev].reset_index()
            if not os.path.exists(crops_folder + prev):
                os.makedirs(crops_folder + prev)
            Parallel(n_jobs=8)(delayed(crop_object_in_field)(
                ix, arr, objects_df, crops_folder + prev, asinh) for ix in range(
                objects_df.shape[0]))
            arr = np.zeros((s0, s1, n_channels), dtype=np.float32)
            if calibrate:
                zps = get_zps(field_name)
            prev = field_name
            i = 0
        data = fits.getdata(f)
        if calibrate:
            data = make_calibration(data, zps[bands[i]])
        arr[:, :, bands_order[i]] = np.copy(data)
        i += 1


def get_min_max(filefolder, n_channels=12):
    """
        receives:
            * filefolder    (str) folder pattern wherein ndarray images are
            * n_channels    (int) number of channels in images
        returns:
            a tuple (minima, maxima), each an array of length=n_channels
                    containing minima and maxima per band across all images
    """
    start = time()
    files = glob(filefolder)
    minima, maxima = np.zeros(n_channels), np.zeros(n_channels)
    min_files, max_files = [''] * n_channels, [''] * n_channels
    n_files = len(files)
    print('nr of files', n_files)
    for file in files:
        im = np.load(file)
        min_tmp = np.min(im, axis=(0, 1))
        max_tmp = np.max(im, axis=(0, 1))

        msk = np.less(min_tmp, minima)
        if msk.any():
            minima[msk] = min_tmp[msk]
            min_files = [file if msk[i] else min_files[i] for i in range(n_channels)]

        msk = np.greater(max_tmp, maxima)
        if msk.any():
            maxima[msk] = max_tmp[msk]
            max_files = [file if msk[i] else max_files[i] for i in range(n_channels)]

    print('minutes taken:', int((time() - start) / 60))
    print('minima', minima)
    print(min_files)
    print('maxima', maxima)
    print(max_files)

    return np.floor(minima), np.ceil(maxima)


def get_mean_var(filefolder, n_channels=12):
    """
        receives:
            * filefolder    (str) folder pattern wherein ndarray images are
            * n_channels    (int) number of channels in images
        returns:
            a tuple (mean, var), each an array of length=n_channels
                    containing mean and variance per band across all images
        reference:
            https://www.researchgate.net/post/
            How_to_combine_standard_deviations_for_three_groups
    """
    start = time()
    files = glob(filefolder)
    mean, var = np.zeros(n_channels), np.zeros(n_channels)
    n_files = len(files)
    print('nr of files', n_files)
    for file in files:
        im = np.load(file)
        mean = mean + np.mean(im, axis=(0, 1))
        var = var + np.std(im, axis=(0, 1))

    mean = mean / n_files
    var = var / n_files

    print('minutes taken:', int((time() - start) / 60))
    print('means', mean)
    print('variances', var)

    return mean, var


def normalize_images(input_folder, output_folder, bounds_lower, bounds_upper):
    """
    saves ndarray images resized to (32,32,n_channels) and normalized to [0,1]
    receives:
        * input_folder      (str) folder path wherein are (x,x,n_channels)
                            ndarray images with varying shapes and value ranges
        * output_folder     (str) folder wherein normalized images will be saved
        * bounds_lower      (ndarray) (n_channels,) array that gives lower
                            bounds for normalization
        * bounds_upper      (ndarray) (n_channels,) array that gives upper
                            bounds for normalization
    """
    files = glob(input_folder)
    print('nr of files', len(files))

    interval = bounds_upper - bounds_lower
    interval = interval[None, None, :]
    lower = bounds_lower[None, None, :]

    start = time()
    for file in files:
        im = np.load(file)
        im = im - lower
        im = im / interval
        if im.min() < 0 or im.max() > 1:
            print('{} out of [0,1] range'.format(file.split('/')[-1]))
        np.save('{}{}'.format(output_folder, file.split('/')[-1]), im)
    print('minutes taken:', int((time() - start) / 60))


if __name__ == '__main__':
    data_dir = os.environ['DATA_PATH']

    sweep_fields(
        fields_path=data_dir + '/dr1/coadded/*/*.fz',
        catalog_path='datasets/clf.csv',
        crops_folder=data_dir + '/crops_calib/',
        calibrate=True,
        asinh=False
    )

    crop_objects_in_rgb(
        catalog_path='datasets/clf.csv',
        input_folder=data_dir + '/dr1/color_images/',
        save_folder=data_dir + '/crops_rgb32/'
    )
