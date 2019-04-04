from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import ascii
from astroquery.sdss import SDSS
import pandas as pd
import time


# SDSS DR15 schema: http://skyserver.sdss.org/dr15/en/help/browser/browser.aspx
# default release for astroquery is 14
# (see astroquery/sdss/__init__.py for configs)

# type: GALAXY (3), STAR (6)
# in SpecObj table: bestObjID = Object ID of photoObj match (position-based)

# Petrosian radius vs effective radius
# https://ryanhausen.github.io/galaxy-classification/2017/10/04/measuring-the-effective-radius-using-the-petrosian-radius.html

# see if this works:
# arcsec per pixel = object size (arcsec) / object size (fwhm pixels)


def query_and_save(query, filename):
	objid = -1
	cnt = 0
	row_count = 500000

	print("querying", filename)
	while row_count==500000:
		start = time.time()
		print("query number", cnt)
		table = SDSS.query_sql(query.format(objid), timeout=600, data_release=15)
		print("seconds taken:", int(time.time()-start))

		row_count = len(table)
		# objid = table[row_count-1]['objID']
		print("row_count", row_count)
		print(table[:10])

		#ascii.write(table, filename.format(cnt), format='csv', fast_writer=False)
		#print("saved to csv")
		cnt+=1


# query objects from sdss

# photo_query = """
# select
# objID, ra, dec, type, probPSF, flags, 
# petroRad_u, petroRad_g, petroRad_r, petroRad_i, petroRad_z, 
# petroRadErr_u, petroRadErr_g, petroRadErr_r, petroRadErr_i, petroRadErr_z
# from PhotoObj
# where abs(ra) < 60 and abs(dec) < 1.25 and objID>{}
# order by objID
# """

# spec_query = """
# select 
# bestObjID, ra, dec, class, subclass
# from SpecObj
# where abs(ra) < 60 and abs(dec) < 1.25
# """

# query_and_save(photo_query, "sdss_photo_{}.csv")
# query_and_save(spec_query, "sdss_spec.csv")


# do catalog match

splus_cat = pd.read_csv('splus_catalog.csv')
splus_ra = splus_cat['ra'].values
splus_dec = splus_cat['dec'].values

sdss_cat = pd.read_csv('sdss_spec.csv')
sdss_ra = sdss_cat['ra'].values
sdss_dec = sdss_cat['dec'].values

my_coord = SkyCoord(ra=sdss_ra*u.degree, dec=sdss_dec*u.degree)
catalog_coord = SkyCoord(ra=splus_ra*u.degree, dec=splus_dec*u.degree)

# a k-d tree is built from catalog_coord (splus)
# my_coord (sdss) is queried on the k-d tree
idx, d2d, d3d = my_coord.match_to_catalog_sky(catalog_coord)  

max_sep = 3.0 * u.arcsec
sep_constraint = d2d < max_sep
my_matches = my_coord[sep_constraint] 
catalog_matches = catalog_coord[idx[sep_constraint]] 

print(my_coord.shape)
print(my_matches.shape)

sdss_cat['splus_idx'] = idx
sdss_cat['d2d'] = d2d
sdss_cat['matched'] = sep_constraint

final_cat = sdss_cat.merge(splus_cat, left_on='splus_idx', right_index=True)

final_cat.to_csv('matched_cat.csv')