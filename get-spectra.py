from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astroquery.sdss import SDSS
import numpy as np
import pandas as pd
import sys

length = 3870

# nohup python3 -u get-spectra.py &
# nohup python3 -u get-spectra.py > /dev/null 2>&1 & echo $! > run.pid

# get object id to start downloading from
f = open('spectra/last_obj.txt')
i = int(f.readline())
f.close()

# get matched object samples
cat = pd.read_csv('matched_cat.csv')
cat = cat[cat.matched].reset_index()
# cat['id'].to_csv('spectra/object_idx.csv')

# load sky coordinates
coords = SkyCoord(
    ra=cat['ra_x'].values*u.degree,
    dec=cat['dec_x'].values*u.degree
    )

object_id = cat['id'].values

del cat

# flux_file = open('spectra/fluxes.dat','ab')
# loglamb_file = open('spectra/loglamb.dat','ab')
# exception_file = open('spectra/exception_ids.dat', 'ab')
# no_match_file = open('spectra/no_match_ids.dat', 'ab')

# download and store flux and wavelength data in text files
for coord in coords[i:]:
    try:
        spec = SDSS.get_spectra(coord, data_release=15, timeout=600)

        # no matches
        if spec is None:
            print('NONE:', i, object_id[i])
            # np.savetxt(no_match_file, np.array([i]))
            # np.savetxt(flux_file, np.zeros((1,length)))
            # np.savetxt(loglamb_file, np.zeros((1,length)))
        # match
        else:
            print('MATCH:', i, object_id[i])
            specdata = spec[0][1].data
            flux = specdata['flux']
            loglamb = specdata['loglam']
            # with open('spectra/flux/{}.dat'.format(object_id[i]), 'xb') as f:
            np.savetxt('spectra/flux/{}.dat'.format(object_id[i]), np.pad(flux, (0, length-flux.shape[0]), 'constant').reshape((1,length)))
            # with open('spectra/loglamb/{}.dat'.format(object_id[i]), 'xb') as f:
            np.savetxt('spectra/loglamb/{}.dat'.format(object_id[i]), np.pad(loglamb, (0, length-loglamb.shape[0]), 'constant').reshape((1,length)))

        if i%100 == 0:
            print('downloaded {}%'.format(100*i/len(coords)))
        i+=1

    # connection error
    except Exception as e:
        print(e)
        print('EXCEPTION:', i, object_id[i])
        # np.savetxt(exception_file, np.array([i]))
        # np.savetxt(flux_file, np.zeros((1,length)))
        # np.savetxt(loglamb_file, np.zeros((1,length)))
        i+=1

'''
each hdul = spectra[i] is an HDUList object
hdul[1].data contains fluxes per each wavelength
hdul[3] (SPZLINE) contains emission lines

available columns for spzline are:
https://classic.sdss.org/dr7/dm/flatFiles/spZline.html

to build a spectrum from data in hdul[1]:
https://specutils.readthedocs.io/en/latest/
'''