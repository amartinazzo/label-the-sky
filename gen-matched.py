from astropy.io import fits
import cv2
from glob import glob
import pandas as pd
import numpy as np

catalog = pd.read_csv("matched_cat.csv")
catalog = catalog[catalog.matched]
catalog = catalog[['id', 'class', 'subclass', 'x', 'y', 'fwhm']]
# catalog['y'] = 11000 - catalog.y
catalog.sort_values(by='id',inplace=True)
catalog.to_csv("sloan_splus_matches.csv", index=False)