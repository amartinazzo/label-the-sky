from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import ascii
from astroquery.sdss import SDSS
from glob import glob
import numpy as np
import pandas as pd
import time

usecols = [
    'ID', 'RA', 'Dec', 'X', 'Y', 'MUMAX', 's2nDet', 'FWHM',
    'uJAVA_auto', 'F378_auto', 'F395_auto', 'F410_auto', 'F430_auto', 'g_auto',
    'F515_auto', 'r_auto', 'F660_auto', 'i_auto', 'F861_auto', 'z_auto',
    'CLASS'
]

usecols_str = "id,ra,dec,x,y,mumax,s2n,fwhm,u,f378,f395,f410,f430,g,f515,r,f660,i,f861,z,class_rf\n"

cols = [
    'id', 'ra', 'dec', 'x', 'y', 'mumax', 's2n', 'fwhm', 
    'u', 'f378', 'f395', 'f410', 'f430', 'g', 'f515', 'r', 'f660', 'i', 'f861', 'z',
    'class', 'subclass'
]


def gen_master_catalog(catalogs_path, output_file, header_file='csv/fits_header_cols.txt'):
    '''
    generates a master catalog from a folder of multiple catalogs (one per field)
    '''
    files = glob(catalogs_path)
    files.sort()
    n_files = len(files)

    # get original cols from txt file
    with open(header_file, 'r') as f:
        cols = f.read().split(',')
    
    with open(output_file, 'w') as f:
        f.write(usecols_str)

    for ix, file in enumerate(files):
        print('{}/{} processing {}...'.format(ix+1, n_files, file))
        # stripe = filename.split('/')[-1].split('.')[0]

        cat = pd.read_csv(file,
            delimiter=' ', skipinitialspace=True, comment='#', index_col=False,
            header=None, names=cols, usecols=usecols)
        cat.dropna(inplace=True)
        int_cols = ['x', 'x', 'class_rf']
        cat[int_cols] = cat[int_cols].apply(lambda x: round(x)).astype(int)
        cat['id'] = cat['id'].apply(lambda s: s.replace('.griz', ''))

        cat.to_csv(output_file, index=False, header=False, mode='a')


def filter_master_catalog(master_cat_file, output_file):
    '''
    generates a catalog from master_catalog_dr_march2019.cat
    filtered by given columns
    '''
    with open(output_file, 'w') as f:
        f.write(usecols_str)

    cat = pd.read_csv(
        master_cat_file, delimiter=' ', skipinitialspace=True, comment='#',
        index_col=False, usecols=usecols)

    cat.dropna(inplace=True)
    int_cols = ['X', 'Y', 'CLASS']
    cat[int_cols] = cat[int_cols].apply(lambda x: round(x)).astype(int)
    cat = cat[usecols]
    cat['ID'] = cat['ID'].apply(lambda s: s.replace('.griz', ''))

    cat.to_csv(output_file, index=False, header=False, mode='a')


def query_sdss(query_str, filename):
    objid = -1
    cnt = 0
    row_count = 500000

    print('querying', filename)
    while row_count==500000:
        start = time.time()
        print('query number', cnt)
        table = SDSS.query_sql(query_str.format(objid), timeout=600, data_release=15)
        print('seconds taken:', int(time.time()-start))

        row_count = len(table)
        objid = table[row_count-1]['objID']
        print('row_count', row_count)
        print(table[:10])

        ascii.write(table, filename.format(cnt), format='csv', fast_writer=False)
        print('saved to csv')
        cnt+=1


def match_catalogs(cat_path, base_cat_path, matched_cat_path, max_distance=2.0):
    print('matching {} on {}'.format(cat_path, base_cat_path))
    base_cat = pd.read_csv(base_cat_path)
    base_ra = base_cat['ra'].values
    base_dec = base_cat['dec'].values

    new_cat = pd.read_csv(cat_path)
    new_ra = new_cat['ra'].values
    new_dec = new_cat['dec'].values

    my_coord = SkyCoord(ra=new_ra*u.degree, dec=new_dec*u.degree)
    catalog_coord = SkyCoord(ra=base_ra*u.degree, dec=base_dec*u.degree)

    # a k-d tree is built from catalog_coord (SPLUS)
    # my_coord (SLOAN) is queried on the k-d tree
    idx, d2d, d3d = my_coord.match_to_catalog_sky(catalog_coord)  

    sep_constraint = d2d < max_distance * u.arcsec
    my_matches = my_coord[sep_constraint] 
    catalog_matches = catalog_coord[idx[sep_constraint]] 

    print(my_coord.shape)
    print(my_matches.shape)

    new_cat['splus_idx'] = idx
    new_cat['d2d'] = d2d
    new_cat['matched'] = sep_constraint

    final_cat = new_cat.merge(base_cat, left_on='splus_idx', right_index=True, suffixes=('_x', ''))
    final_cat.to_csv(matched_cat_path, index=False)

    return final_cat


def gen_filtered_catalog(cat, output_file, spectra_folder=None):
    c = pd.read_csv(cat)
    shape_prev = c.shape[0]
    print('original shape', shape_prev)
    c = c[c.matched]
    print('shape/diff after filtering matched', c.shape[0], shape_prev-c.shape[0])

    shape_prev = c.shape[0]
    duplicates = c['id'].value_counts()
    duplicates = duplicates[duplicates>1].index.values
    c = c[~c['id'].isin(duplicates)]
    print('shape/diff after removing duplicates', c.shape[0], shape_prev-c.shape[0])

    c['id'] = c['id'].apply(lambda s: s.replace('.griz', ''))
    c = c[cols]

    if spectra_folder is not None:
        spectra = glob(spectra_folder)
        for i, s in enumerate(spectra):
            spectra[i] = s.split('/')[-1][:-4]
        shape_prev = c.shape[0]
        c = c[c['id'].isin(spectra)]
        print('shape/diff after removing objects without corresponding spectrum',
            c.shape[0], shape_prev-c.shape[0])

    c.sort_values(by='id',inplace=True)
    c.to_csv(output_file, index=False)


def gen_splits(df_filename, val_split=0.1, test_split=0.1):
    df = pd.read_csv(df_filename)
    np.random.seed(0)

    msk = np.random.rand(len(df)) > test_split
    df_trainval = df[msk]
    df_test = df[~msk]
    del df

    msk = np.random.rand(len(df_trainval)) > val_split
    df_train = df_trainval[msk]
    df_val = df_trainval[~msk]
    del df_trainval

    base_filename = df_filename.split('.')[0]

    df_train.to_csv(base_filename + '_train.csv', index=False)
    df_val.to_csv(base_filename + '_val.csv', index=False)
    df_test.to_csv(base_filename + '_test.csv', index=False)

    print('train set:', df_train.shape[0])
    print(df_train['class'].value_counts(normalize=True))
    print('val set:', df_val.shape[0])
    print(df_val['class'].value_counts(normalize=True))
    print('test set:', df_test.shape[0])
    print(df_test['class'].value_counts(normalize=True))


def verify_downloaded_spectra(cat, output_file, spectra_folder):
    c = pd.read_csv(cat)



if __name__=='__main__':
    # query objects from sdss

    # photo_query = '
    # select
    # objID, ra, dec, type, probPSF, flags, 
    # petroRad_u, petroRad_g, petroRad_r, petroRad_i, petroRad_z, 
    # petroRadErr_u, petroRadErr_g, petroRadErr_r, petroRadErr_i, petroRadErr_z
    # from PhotoObj
    # where abs(ra) < 60 and abs(dec) < 1.25 and objID>{}
    # order by objID
    # '
    # spec_query = '
    # select 
    # bestObjID, ra, dec, class, subclass
    # from SpecObj
    # where abs(ra) < 60 and abs(dec) < 1.25
    # '
    # query_sdss(photo_query, 'sdss_photo_{}.csv')
    # query_sdss(spec_query, 'csv/sdss_spec.csv')

    splus_cat = 'csv/splus_catalog_dr1.csv'
    sloan_cat = 'csv/sdss_spec.csv'
    matched_cat ='csv/matched_cat_dr1.csv'
    filtered_cat = 'csv/matched_cat_dr1_filtered.csv'

    # generate master catalog
    print('generating master splus catalog')
    start = time.time()
    filter_master_catalog('../raw-data/dr1/SPLUS_STRIPE82_master_catalog_dr_march2019.cat', splus_cat)
    print('minutes taken:', int((time.time()-start)/60))

    # match catalogs
    print('matching splus and sloan catalogs')
    start = time.time()
    c = match_catalogs(sloan_cat, splus_cat, matched_cat)
    print('minutes taken:', int((time.time()-start)/60))

    # filter catalog
    print('filtering matched catalog')
    gen_filtered_catalog(matched_cat, filtered_cat)

    print('generating train-val-test splits')
    gen_splits(filtered_cat)