from glob import glob
import numpy as np
import pandas as pd
import os

base_path = os.path.join(os.environ['DATA_PATH'], 'crops_calib')
csv_file = os.environ['HOME']+'/label_the_sky/csv/dr1_unlabeled.csv'

def get_minmax(idx):
	im = np.load(os.path.join(base_path, idx.split('.')[0], idx+'.npy'))
	return im.min(), im.max()

df = pd.read_csv(csv_file)

df['minmax'] = df.id.apply(get_minmax)
df['min'] = df.minmax.apply(lambda d: d[0])
df['max'] = df.minmax.apply(lambda d: d[1])
df.drop(columns='minmax', inplace=True)

df.to_csv(csv_file, index=False)