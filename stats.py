from gen_catalog import stratified_split
import numpy as np
import pandas as pd
import random


df = pd.read_csv("csv/dr1_crossmatched_modelmags.csv")

df['filename'] = df[['plate', 'mjd', 'fiberID']].apply(
    lambda x: "spec-{}-{}-{}".format(str(x[0]).zfill(4), str(x[1]), str(x[2]).zfill(4)),
    axis=1
)

df_mocks = pd.read_csv("csv/dr1_mocks.csv")

# shuffled = list(df.filename.values)
# random.shuffle(shuffled)
# df_mocks["filename"] = shuffled

df = df.merge(df_mocks, on="filename", how="left", suffixes=('', '_mock'))
print('shape before removing 99.', df.shape)

df.dropna(inplace=True)
df = df[~((df==99.).values.any(axis=1))]
print('shape after removing 99.', df.shape)

# check that ill objects were all removed
assert (df.photoflag==0).sum() == df.shape[0]
assert (df.ndet==12).sum() == df.shape[0]
assert (df.zWarning==0).sum() == df.shape[0]

df = stratified_split(df, e=0.1)
df.drop(columns=['filename'], inplace=True)
df.to_csv("csv/dr1_split.csv", index=False)

for b in ['u','f378','f395','f410','f430','g','f515','r','f660','i','f861','z']:
    df[f'{b}_diff'] = df[b] - df[f'{b}_mock']

stats = df[[
    'u','f378','f395','f410','f430','g','f515','r','f660','i','f861','z',
    'u_mock','f378_mock','f395_mock','f410_mock','f430_mock','g_mock','f515_mock','r_mock','f660_mock','i_mock','f861_mock','z_mock',
    'u_diff','f378_diff','f395_diff','f410_diff','f430_diff','g_diff','f515_diff','r_diff','f660_diff','i_diff','f861_diff','z_diff',
]].describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']].round(4)

print(stats)

# check outliers
# thres = 5
# large_diffs = (np.abs(df.u_diff)>thres) | (np.abs(df.f378_diff)>thres) | (np.abs(df.f395_diff)>thres) | (np.abs(df.f410_diff)>thres) | (
#     np.abs(df.f430_diff)>thres) |(np.abs(df.g_diff)>thres) | (np.abs(df.f515_diff)>thres) | (np.abs(df.r_diff)>thres) | (
#     np.abs(df.f660_diff)>thres) | (np.abs(df.i_diff)>thres) | (np.abs(df.f861_diff)>thres) | (np.abs(df.z_diff)>thres)

# outliers = df.loc[large_diffs, [
#     'u','f378','f395','f410','f430','g','f515','r','f660','i','f861','z',
#     'u_mock','f378_mock','f395_mock','f410_mock','f430_mock','g_mock','f515_mock','r_mock','f660_mock','i_mock','f861_mock','z_mock',
#     'u_diff','f378_diff','f395_diff','f410_diff','f430_diff','g_diff','f515_diff','r_diff','f660_diff','i_diff','f861_diff','z_diff',
# ]]

# print(outliers)