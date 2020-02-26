from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import ascii
from astroquery.sdss import SDSS
from glob import glob
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import time


orig_cols = [
    'ID', 'RA', 'Dec', 'X', 'Y', 'ISOarea', 's2nDet', 'PhotoFlag', 'FWHM', 'MUMAX', 'A', 'B', 'THETA', 'FlRadDet', 'KrRadDet',
    'uJAVA_auto','euJAVA_auto', 's2n_uJAVA_auto', 'uJAVA_petro', 'euJAVA_petro', 's2n_uJAVA_petro', 'uJAVA_aper', 'euJAVA_aper', 's2n_uJAVA_aper',
    'F378_auto', 'eF378_auto', 's2n_F378_auto', 'F378_petro', 'eF378_petro', 's2n_F378_petro', 'F378_aper', 'eF378_aper', 's2n_F378_aper',
    'F395_auto','eF395_auto', 's2n_F395_auto', 'F395_petro', 'eF395_petro', 's2n_F395_petro', 'F395_aper', 'eF395_aper', 's2n_F395_aper',
    'F410_auto', 'eF410_auto', 's2n_F410_auto', 'F410_petro', 'eF410_petro', 's2n_F410_petro', 'F410_aper', 'eF410_aper', 's2n_F410_aper',
    'F430_auto', 'eF430_auto', 's2n_F430_auto', 'F430_petro', 'eF430_petro', 's2n_F430_petro', 'F430_aper', 'eF430_aper', 's2n_F430_aper',
    'g_auto', 'eg_auto', 's2n_g_auto', 'g_petro', 'eg_petro', 's2n_g_petro', 'g_aper', 'eg_aper', 's2n_g_aper',
    'F515_auto', 'eF515_auto', 's2n_F515_auto', 'F515_petro', 'eF515_petro', 's2n_F515_petro', 'F515_aper', 'eF515_aper', 's2n_F515_aper',
    'r_auto', 'er_auto', 's2n_r_auto', 'r_petro', 'er_petro', 's2n_r_petro', 'r_aper', 'er_aper', 's2n_r_aper',
    'F660_auto', 'eF660_auto', 's2n_F660_auto', 'F660_petro', 'eF660_petro', 's2n_F660_petro', 'F660_aper', 'eF660_aper', 's2n_F660_aper',
    'i_auto', 'ei_auto', 's2n_i_auto', 'i_petro', 'ei_petro', 's2n_i_petro', 'i_aper', 'ei_aper', 's2n_i_aper',
    'F861_auto', 'eF861_auto', 's2n_F861_auto', 'F861_petro', 'eF861_petro', 's2n_F861_petro', 'F861_aper', 'eF861_aper', 's2n_F861_aper',
    'z_auto', 'ez_auto', 's2n_z_auto', 'z_petro', 'ez_petro', 's2n_z_petro', 'z_aper', 'ez_aper', 's2n_z_aper',
    'zb', 'zb_Min', 'zb_Max', 'Tb', 'Odds', 'Chi2', 'M_B', 'Stell_Mass', 'CLASS', 'PROB_GAL', 'PROB_STAR'
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
            header=None, names=orig_cols, usecols=usecols)

        cat.dropna(inplace=True)
        int_cols = ['X', 'Y']
        cat[int_cols] = cat[int_cols].apply(lambda x: round(x)).astype(int)
        cat['ID'] = cat['ID'].apply(lambda s: s.replace('.griz', ''))

        cat.to_csv(output_file, index=False, header=False, mode='a')


def filter_master_catalog(master_cat_file, output_file, usecols_orig, usecols_renamed):
    '''
    generates a catalog from master_catalog_dr_march2019.cat
    filtered by given columns
    '''
    cat = pd.read_csv(
        master_cat_file, delimiter=' ', skipinitialspace=True, comment='#',
        index_col=False, usecols=usecols)

    cat.dropna(inplace=True)
    int_cols = ['X', 'Y']
    cat[int_cols] = cat[int_cols].apply(lambda x: round(x)).astype(int)
    cat = cat[usecols_orig]
    cat.columns = usecols_renamed
    cat['id'] = cat.id.str.replace('.griz', '')
    cat['id'] = cat.id.str.replace('SPLUS.', '')

    cat.to_csv(output_file, index=False)


# m = 22.5 - 2.5*log10(FLUX)

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
        objid = table[row_count-1]['bestObjID']
        print('row_count', row_count)
        print(table[:10])

        ascii.write(table, filename.format(cnt), format='csv', fast_writer=False)
        print('saved to csv')
        cnt+=1


def match_catalogs(new_df, base_df, matched_cat_path=None, max_distance=1.0):
    print('matching')
    base_ra = base_df['ra'].values
    base_dec = base_df['dec'].values

    new_ra = new_df['ra'].values
    new_dec = new_df['dec'].values

    my_coord = SkyCoord(ra=new_ra*u.degree, dec=new_dec*u.degree)
    catalog_coord = SkyCoord(ra=base_ra*u.degree, dec=base_dec*u.degree)

    # a k-d tree is built from catalog_coord (SPLUS)
    # my_coord (SLOAN) is queried on the k-d tree
    idx, d2d, d3d = my_coord.match_to_catalog_sky(catalog_coord)

    print('len unique idx', len(np.unique(idx)))
    print('len idx', len(idx))

    sep_constraint = d2d < max_distance * u.arcsec
    my_matches = my_coord[sep_constraint] 
    catalog_matches = catalog_coord[idx[sep_constraint]] 

    print('my coord matches', my_matches.shape)
    print('catalog matches', catalog_matches.shape)

    new_df['base_idx'] = idx
    new_df['d2d'] = d2d
    new_df['matched'] = sep_constraint
    new_df = new_df[new_df.matched]
    print(new_df.base_idx.value_counts())
    # agg = new_df.groupby('base_idx')
    # agg = agg.apply(lambda s: s.sort_values('d2d', ascending=True).head(1))
    # agg.drop(columns=['base_idx'], inplace=True)
    # print(agg.d2d.min(), agg.d2d.max())

    final_cat = base_df.merge(new_df, how='left', left_index=True, right_on='base_idx', suffixes=('', '_'))
    final_cat = final_cat[~final_cat.matched.isna()]

    final_cat['redshift'] = final_cat.z_

    final_cat = final_cat[[
        'id','ra','dec','class','subclass','x','y','mumax','s2n','photoflag','ndet','fwhm',
        'u','f378','f395','f410','f430','g','f515','r','f660','i','f861','z',
        'spectroFlux_u', 'spectroFlux_g', 'spectroFlux_r', 'spectroFlux_i', 'spectroFlux_z', 
        'd2d','redshift',
        'u_err','f378_err','f395_err','f410_err','f430_err','g_err',
        'f515_err','r_err','f660_err','i_err','f861_err','z_err']]

    print('matched df shape', new_df.shape)
    print('base df shape', base_df.shape)
    print('final df shape', final_cat.shape)
    
    if matched_cat_path is not None:
        final_cat.to_csv(matched_cat_path, index=False)

    return final_cat


def add_downloaded_spectra_col(cat, spectra_folder):
    spectra = glob(spectra_folder)
    spectra = [s.split('/')[-1][:-4] for s in spectra]
    print(spectra[:10])
    c = pd.read_csv(cat)
    c['has_spectra'] = c['id'].apply(lambda s: s in spectra)
    c.to_csv(cat, index=False)


def pixels_to_int(filepath):
    df = pd.read_csv(filepath)
    df['x'] = df.x.astype(int)
    df['y'] = df.y.astype(int)
    df.to_csv(filepath, index=False)


def fill_undetected(df):
    mags = ['u','f378','f395','f410','f430','g','f515','r','f660','i','f861','z']
    df_mags = df[mags]
    df_mags[df_mags.values==99] = np.nan
    df_mags[df_mags.values==-99] = np.nan
    df_mags = df_mags.apply(lambda row: row.fillna(row.median()), axis=1) # fill with median per row
    df[mags] = df_mags.values


def stratified_split(filepath, mag_range=None, fill_undet=False, test_split=0.1, val_split=0.11, e=None):
    df = pd.read_csv(filepath)
    df = df[(~df['class'].isna()) & (df.ndet==12) & (df.photoflag==0)]
    if e is not None:
        df = df[(
            df.u_err <= e) & (df.f378_err <= e) & (df.f395_err <= e) & (df.f410_err <= e) & (df.f430_err <= e) & (df.g_err <= e) & (
            df.f515_err <= e) & (df.r_err <= e) & (df.f660_err <= e) & (df.i_err <= e) & (df.f861_err <= e) & (df.z_err <= e)]
    if mag_range is not None:
        df = df[df.r.between(mag_range[0], mag_range[1])]

    print('shape after filtering', df.shape)

    if fill_undet:
        print('filling undetected')
        fill_undetected(df)

    df.loc[df.r<14,'r'] = 14
    df.loc[df.r>21,'r'] = 21
    df['class_mag'] = np.round(df.r.values).astype(np.uint8)
    df.loc[df['class']=='QSO', 'class_mag'] = df.class_mag.apply(lambda r: r if r%2==0 else r+1)
    df['class_mag'] = df['class'] + df['class_mag'].astype(str)
    
    # hard code small subsets
    df.loc[df.class_mag=='QSO14', 'class_mag'] = 'QSO16'
    df.loc[df.class_mag=='GALAXY24', 'class_mag'] = 'GALAXY23'
    df.loc[df.class_mag=='STAR23', 'class_mag'] = 'STAR22'

    df['class_mag'] = df['class_mag'].astype('category')
    print(df.class_mag.value_counts(normalize=False))
    df['class_mag_int'] = df.class_mag.cat.codes
    df['split'] = ''

    # train-test split
    X = df.index.values
    y = df.class_mag_int.values
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=0)
    train_idx, test_idx = next(sss.split(X,y))
    df_train_idx, df_test_idx = X[train_idx], X[test_idx]
    df.loc[df_train_idx, 'split'] = 'train'
    df.loc[df_test_idx, 'split'] = 'test'

    # train-val split
    X = df[df.split=='train'].index.values
    y = df[df.split=='train'].class_mag_int.values
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=0)
    train_idx, test_idx = next(sss.split(X,y))
    df_train_idx, df_test_idx = X[train_idx], X[test_idx]
    df.loc[df_train_idx, 'split'] = 'train'
    df.loc[df_test_idx, 'split'] = 'val'

    print(df.split.value_counts(normalize=True))
    print()

    print('SPLITS PER CAT')
    print()
    for i in np.unique(df.class_mag_int.values):
        dff = df[df.class_mag_int==i]
        print(dff['class_mag'].head(1))
        print(dff.split.value_counts(normalize=True))
        print()

    df.drop(columns=['class_mag', 'class_mag_int'], inplace=True)
    df.to_csv('{}_split.csv'.format(filepath[:-4]), index=False)


def stratified_split_unlabeled(filepath, mag_range=None, test_split=0.1, val_split=0.11):
    df = pd.read_csv(filepath)
    s0 = df.shape
    print(s0)
    df = df[df['min']!=df['max']]
    print(df.shape)
    print(s0[0]-df.shape[0])
    if mag_range is not None:
        df = df[df.r.between(mag_range[0], mag_range[1])]
    df['class_mag'] = np.round(df.r.values).astype(np.uint8)

    # train-test split
    X = df.index.values
    y = df.class_mag.values
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=0)
    train_idx, test_idx = next(sss.split(X,y))
    df_train_idx, df_test_idx = X[train_idx], X[test_idx]
    df.loc[df_train_idx, 'split'] = 'train'
    df.loc[df_test_idx, 'split'] = 'test'

    # train-val split
    X = df[df.split=='train'].index.values
    y = df[df.split=='train'].class_mag.values
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=0)
    train_idx, test_idx = next(sss.split(X,y))
    df_train_idx, df_test_idx = X[train_idx], X[test_idx]
    df.loc[df_train_idx, 'split'] = 'train'
    df.loc[df_test_idx, 'split'] = 'val'

    df.drop(columns=['class_mag'], inplace=True)
    df.to_csv('{}_split.csv'.format(filepath[:-4]), index=False)


if __name__=='__main__':
    # query objects from sdss

    photo_query = '''
    select
    objID, ra, dec, type, probPSF, flags, 
    petroRad_u, petroRad_g, petroRad_r, petroRad_i, petroRad_z, 
    petroRadErr_u, petroRadErr_g, petroRadErr_r, petroRadErr_i, petroRadErr_z
    from PhotoObj
    where abs(dec) < 1.25 and objID>{}
    order by objID
    '''

    spec_query = '''
    select 
    bestObjID, ra, dec, class, subclass, z, zErr, zWarning,
    spectroFlux_u, spectroFlux_g, spectroFlux_r, spectroFlux_i, spectroFlux_z,
    spectroFluxIvar_u, spectroFluxIvar_g, spectroFluxIvar_r, spectroFluxIvar_i, spectroFluxIvar_z
    from SpecObj
    where abs(dec) < 1.46
    '''
    # master catalog: dec in (-1.4139, 1.4503)
    
    # query_sdss(spec_query, 'csv/sdss_spec_full_STRIPE82.csv')

    # gen master catalog
    data_dir = os.environ['DATA_PATH']

    usecols = [
        'ID', 'RA', 'Dec', 'X', 'Y', 'MUMAX', 's2nDet', 'PhotoFlag', 'nDet_auto', 'FWHM',
        'uJAVA_auto', 'F378_auto', 'F395_auto', 'F410_auto', 'F430_auto', 'g_auto',
        'F515_auto', 'r_auto', 'F660_auto', 'i_auto', 'F861_auto', 'z_auto',
        'euJAVA_auto','eF378_auto','eF395_auto','eF410_auto','eF430_auto','eg_auto',
        'eF515_auto','er_auto','eF660_auto','ei_auto','eF861_auto','ez_auto'
    ]

    usecols_renamed = [
        'id', 'ra', 'dec', 'x', 'y', 'mumax', 's2n', 'photoflag', 'ndet', 'fwhm', 
        'u', 'f378', 'f395', 'f410', 'f430', 'g', 'f515', 'r', 'f660', 'i', 'f861', 'z',
        'u_err','f378_err','f395_err','f410_err','f430_err','g_err',
        'f515_err','r_err','f660_err','i_err','f861_err','z_err'
    ]

    # filter_master_catalog(data_dir+'/dr1/SPLUS_STRIPE82_master_catalog_dr_march2019.cat', 'csv/dr1.csv', usecols, usecols_renamed)
    
    # match catalogs
    splus_cat = pd.read_csv('csv/dr1.csv')
    sloan_cat = pd.read_csv('csv/sdss_spec_full_STRIPE82.csv')
    matched_cat ='csv/dr1_classes.csv'
    c = match_catalogs(sloan_cat, splus_cat, matched_cat)

    # gen split catalog
    stratified_split(matched_cat)