from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import ascii
from astroquery.sdss import SDSS
from glob import glob
import pandas as pd
import time


def gen_master_catalog(catalogs_path, output_file, header_file='csv/fits_header_cols.txt'):
    files = glob(catalogs_path)
    files.sort()
    n_files = len(files)

    # get original cols from txt file
    with open(header_file, 'r') as f:
        cols = f.read().split(',')

    # choose output cols and write header to master catalog (in the same order)
    usecols = ['ID', 'RA', 'Dec', 'X', 'Y', 'MUMAX', 's2nDet', 'FWHM', 'M_B', 'r_auto', 'CLASS']
    with open(output_file, 'w') as f:
        f.write('id,ra,dec,x,y,mumax,s2n,fwhm,absolute_mag,r_mag_auto,class_rf')

    for ix, file in enumerate(files):
        print('{}/{} processing {}...'.format(ix+1, n_files, file))
        # stripe = filename.split('/')[-1].split('.')[0]

        cat = pd.read_csv(file,
            delimiter=' ', skipinitialspace=True, comment='#', index_col=False,
            header=None, names=cols, usecols=usecols)
        cat.dropna(inplace=True)
        int_cols = ['X', 'Y', 'CLASS']
        cat[int_cols] = cat[int_cols].apply(lambda x: round(x)).astype(int)
        cat['ID'] = cat['ID'].apply(lambda s: s.replace('.griz', ''))

        cat.to_csv(output_file, index=False, header=False, mode='a')


def filter_master_cat(master_cat_file, output_file, header_file='csv/fits_header_cols.txt'):
    # get original cols from txt file
    with open(header_file, 'r') as f:
        cols = f.read().split(',')

    # choose output cols and write header to master catalog (in the same order)
    usecols = ['ID', 'RA', 'Dec', 'X', 'Y', 'MUMAX', 's2nDet', 'FWHM', 'M_B', 'r_auto', 'CLASS']
    with open(output_file, 'w') as f:
        f.write('id,ra,dec,x,y,mumax,s2n,fwhm,absolute_mag,r_mag_auto,class_rf')

    cat = pd.read_csv(
        master_cat_file, delimiter=' ', skipinitialspace=True, comment='#',
        index_col=False, header=None, names=cols, usecols=usecols)
    cat.dropna(inplace=True)
    int_cols = ['X', 'Y', 'CLASS']
    cat[int_cols] = cat[int_cols].apply(lambda x: round(x)).astype(int)
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

    final_cat = new_cat.merge(base_cat, left_on='splus_idx', right_index=True)
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

    if spectra_folder is not None:
        spectra = glob(spectra_folder)
        for i, s in enumerate(spectra):
            spectra[i] = s.split('/')[-1][:-4]
        shape_prev = c.shape[0]
        c = c[c['id'].isin(spectra)]
        print('shape/diff after removing objects without corresponding spectrum',
            c.shape[0], shape_prev-c.shape[0])

    c = c[['id', 'class','x','y','fwhm','r_mag_auto']]
    c.sort_values(by='id',inplace=True)
    c.to_csv(output_file, index=False)


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

    splus_cat = 'csv/splus_catalog_dr1_mag.csv'
    sloan_cat = 'csv/sdss_spec.csv'
    matched_cat ='csv/matched_cat_dr1.csv'
    filtered_cat = 'csv/matched_cat_dr1_filtered.csv'

    # generate master catalog
    print('generating master splus catalog')
    filter_master_cat('../raw-data/dr1/SPLUS_STRIPE82_master_catalog_dr_march2019.cat', splus_cat)

    # match catalogs
    print('matching splus and sloan catalogs')
    start = time.time()
    c = match_catalogs(sloan_cat, splus_cat, matched_cat)
    print('minutes taken:', int((time.time()-start)/60))

    # filter catalog
    print('filtering matched catalog')
    gen_filtered_catalog(matched_cat, filtered_cat)