from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astroquery.sdss import SDSS
import pandas as pd

cat = pd.read_csv('matched_cat.csv')
cat = cat[cat.matched].reset_index()

# 3 is galaxy, 6 is star
print(cat[['class', 'class_rf']].sample(20))

# TODO remove [0] to get spectra for all objects
coords = SkyCoord(
	ra=cat['ra_x'].values[0]*u.degree,
	dec=cat['dec_x'].values[0]*u.degree
	)

spec = SDSS.get_spectra(coords)
spec[0].writeto('spectest.fits')

'''
hdul = spec[0] is an HDUList object
hdul[3] (SPZLINE) contains the emission lines of the astronomical object

available columns are:
https://classic.sdss.org/dr7/dm/flatFiles/spZline.html

which ones would be useful for us? area? width?

https://specutils.readthedocs.io/en/latest/
'''