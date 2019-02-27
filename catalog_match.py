from astropy.io import ascii
from astroquery.sdss import SDSS
import pandas as pd
import time


# SDSS DR15 schema: http://skyserver.sdss.org/dr15/en/help/browser/browser.aspx
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


photo_query = """
select
objID, ra, dec, type, probPSF, flags, 
petroRad_u, petroRad_g, petroRad_r, petroRad_i, petroRad_z, 
petroRadErr_u, petroRadErr_g, petroRadErr_r, petroRadErr_i, petroRadErr_z
from PhotoObj
where abs(ra) < 60 and abs(dec) < 1.25 and objID>{}
order by objID
"""


spec_query = """
select 
bestObjID, ra, dec, class, subclass
from SpecObj
where abs(ra) < 60 and abs(dec) < 1.25
"""

query_and_save(photo_query, "sdss_photo_{}.csv")
# query_and_save(spec_query, "sdss_spec.csv")