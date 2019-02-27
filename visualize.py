from astropy import visualization as vis
from astropy.io import fits
from glob import glob
import matplotlib.pyplot as plt


sqrt_stretcher = vis.SqrtStretch()
scaler = vis.ZScaleInterval()

files = glob("coadded/*", recursive=True)

im_path = files[0]
print(im_path)
fits_im = fits.open(im_path)

data = fits_im[1].data #ndarray

print(data)
print(data.max(), data.min())
data_orig = data

data = sqrt_stretcher(data)
vmin, vmax = scaler.get_limits(data)

print(data)
print(vmin, vmax)
plt.imshow(data, cmap='gray', vmin=vmin, vmax=vmax)
plt.show()
