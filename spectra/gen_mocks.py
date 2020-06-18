"""
adapted from code by Carolina Queiroz
"""

from astropy.io import fits
from glob import glob
import pandas as pd
from multiprocessing import cpu_count, Pool, Manager
import numpy as np
from scipy import interpolate
import sys
import time


if len(sys.argv) != 5:
    print('usage: python {} <input_csv> <output_csv> <filter_folder> <spectra_folder>'.format(sys.argv[0]))
    exit()

input_file = sys.argv[1]
output_file = sys.argv[2]
FILTERS_PATH = sys.argv[3]
SPECTRA_PATH = sys.argv[4]

vel_light = 2.99792458 * 10**18  # angstrom/s


filter_names = ['uJAVA', 'F378', 'F395', 'F410', 'F430', 'gSDSS', 'F515', 'rSDSS', 'F660', 'iSDSS', 'F861', 'zSDSS']
N_FILTERS = len(filter_names)
# central wavelengths (angstroms)
L = np.array(
    [3533.35, 3773.12, 3940.71, 4094.93, 4292.15, 4758.49, 5133.15, 6251.89, 6613.86, 7670.63, 8607.25, 8941.44]
)

wavel = []
wavef = []
for f in filter_names:
    file = np.loadtxt(FILTERS_PATH + f + '.dat', unpack=True)
    wavel.append(10. * file[0])  # wavelengths are in nanometers in the S-PLUS files
    wavef.append(file[1])


###

df = pd.read_csv(input_file)

df['filename'] = df[['plate', 'mjd', 'fiberID']].apply(
    lambda x: "spec-{}-{}-{}".format(str(x[0]).zfill(4), str(x[1]), str(x[2]).zfill(4)),
    axis=1
)

# assertions

# all filenames are unique
assert df.shape[0] == len(df.filename.value_counts())

# all files in the dataframe exist
fits_files = glob(SPECTRA_PATH + '*.fits')
fits_files = [f.split('/')[-1][:-5] for f in fits_files]
assert df.filename.isin(fits_files).sum() == df.shape[0]

"""
Corrections to the filters not fully covered by SDSS spectra
For uJAVA, these were obtained by measuring the median difference
between the S-PLUS mag and the SDSS mag for bright sources (PhotoFlag=0, zWarning=0)
For F378, this was obtained by the median difference between the
observations in F378 and uJAVA
"""

# TODO compute values instead of hardcoding them
corr_u = {
    'GALAXY': 1.36,
    'STAR': 0.38,
    'QSO': 0.34
}

corr_f378 = {
    'GALAXY': -0.3,
    'STAR': -0.5,
    'QSO': -0.32
}

df['corr_u'] = df['class'].apply(lambda c: corr_u[c])
df['corr_f378'] = df['class'].apply(lambda c: corr_f378[c])

filenames = df.filename.values
usdss = df.modelMag_u.values
corr_u = df.corr_u.values
corr_f378 = df.corr_f378.values


# FUNCS

# Convert AB magnitude to fluxes in units of erg/s/cm^2/angstrom
def f_lambda(wavelength, mag_ab):
    value = (vel_light / wavelength**2) * 10**(-0.4 * (mag_ab + 48.6))
    return (value)


def convolve(filename, mag_u_sdss, correction_u, correction_f378, mocks):
    file = SPECTRA_PATH + filename + ".fits"

    data = fits.getdata(file)
    try:
        F = data.flux
        W = data.loglam
        iVar = data.ivar
    except AttributeError:
        F = data.FLUX
        W = data.LOGLAM
        iVar = data.IVAR
    del data

    # saturated pixels are not flagged as such, but have iVar = 0.0
    # this deals with saturated pixels, avoiding artificial emission lines
    W = np.power(10, W)

    F[iVar == 0.0] = 0.0001 * np.average(F)
    iVar[iVar == 0] = 1.0
    EF = 1. / np.sqrt(iVar)
    F[F <= 0.0] = 0.0001 * np.average(F)
    EF[EF <= 0.0] = max(EF)

    f_convol = np.zeros(N_FILTERS)  # erg/s/cm^2/angstrom

    for i in range(N_FILTERS):
        Wlambda = []
        pos = np.where((W >= np.min(wavel[i])) & (W <= np.max(wavel[i])))

        if len(pos[0]) == 0.0:
            f_convol[i] = 0.
        else:
            Wlambda.append([np.min(pos), np.max(pos)])
            Lambda = np.linspace(W[Wlambda[0][0]], W[Wlambda[0][1]], len(pos[0]))
            swavel_interp = interpolate.InterpolatedUnivariateSpline(wavel[i], wavef[i], k=3)

            f_convol[i] = np.sum(
                F[Wlambda[0][0]:Wlambda[0][1]+1]*10**(-17)*swavel_interp(Lambda))/np.sum(swavel_interp(Lambda))

    # Correct the fluxes in uJAVA and F378
    mag_u_splus = mag_u_sdss + correction_u
    mag_378_splus = mag_u_splus + correction_f378

    f_convol[0] = f_lambda(L[0], mag_u_splus)
    f_convol[1] = f_lambda(L[1], mag_378_splus)

    # Convert F_lambda to F_nu to take into account the spec_scale
    f_convol_nu = (L**2 / vel_light) * f_convol  # erg/s/cm^2/Hz
    # Convert F_nu to AB magnitudes
    mag_AB = -2.5 * np.log10(f_convol_nu) - 48.6
    mag_AB = np.round(mag_AB, 5)

    print(filename, end=',')
    for i, m in enumerate(mag_AB):
        end = '\n' if i==len(mag_AB)-1 else ','
        print(m, end=end)

    # mocks[filename] = mag_AB
    # print("processed", filename)


# compute mocks

start = time.time()

# manager = Manager()
# pool = Pool(processes=cpu_count())

mocks = {}

print('filename', end=',')
for i, f in enumerate(filter_names):
    end = '\n' if i==len(filter_names)-1 else ','
    print(f, end=end)

for ix, f in enumerate(filenames):
    convolve(f, usdss[ix], corr_u[ix], corr_f378[ix], mocks)

#     pool.apply_async(
#         convolve,
#         args=(f, usdss[ix], corr_u[ix], corr_f378[ix], mocks)
#     )
# pool.close()
# pool.join()

print("\n\nseconds taken", time.time() - start)


# save to csv

# dff = pd.DataFrame.from_dict(mocks, columns=filter_names, orient='index')
# dff = dff.round(5)
# dff.to_csv(output_file)
