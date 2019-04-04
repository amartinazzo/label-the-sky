from constants import cols
from glob import glob
import pandas as pd

cat_file = "splus_catalog.csv"
files = glob("../raw_data/catalogs/*", recursive=True)
n_files = len(files)

usecols = ['ID', 'RA', 'Dec', 'X', 'Y', 'MUMAX', 's2nDet', 'FWHM', 'CLASS']
#id,ra,dec,x,y,mumax,s2n,fwhm,class_rf

for ix, file in enumerate(files):
    print('{}/{} processing {}...'.format(ix+1, n_files, file))
    # stripe = filename.split('/')[-1].split('.')[0]

    cat = pd.read_csv(file,
        delimiter=' ', skipinitialspace=True, comment='#', index_col=False,
        header=None, names=cols, usecols=usecols)
    cat.dropna(inplace=True)
    int_cols = ['X', 'Y', 'CLASS']
    cat[int_cols] = cat[int_cols].apply(lambda x: round(x)).astype(int)

    cat.to_csv(cat_file, index=False, header=False, mode='a')