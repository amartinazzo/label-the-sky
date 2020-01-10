import pandas as pd
import os
from shutil import copyfile


basepath = os.environ['DATA_PATH']+'/crops_rgb32/'

df = pd.read_csv('csv/dr1_classes_split.csv')
df = df[(df.photoflag==0)&(df.ndet==12)]

for c in ['STAR', 'GALAXY', 'QSO']:
	dff = df[df['class']==c].sample(3)
	ids = dff['id'].values
	for i in ids:
		src = basepath+i.split('.')[0]+'/'+i+'.png'
		dst = i+'_'+c+'.png'
		print('copying', src, dst)
		copyfile(src, dst)
