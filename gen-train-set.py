from glob import glob
import pandas as pd

spectra = glob('../raw-data/spectra/*')
print('nr of spectra', len(spectra))
for i, s in enumerate(spectra):
	spectra[i] = s.split('/')[-1][:-4]
print(spectra[:10])

cat = pd.read_csv('matched_cat.csv')
cat = cat[['id','class']]
print('cat shape', cat.shape)

cat['id'] = cat['id'].apply(lambda s: s.replace('.griz', ''))
cat.sort_values('id', inplace=True)
print(cat.head(5))

cat = cat[cat['id'].isin(spectra)]
print('cat shape after filtering', cat.shape)

cat.to_csv('train_val_set.csv', index=False)