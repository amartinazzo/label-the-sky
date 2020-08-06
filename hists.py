import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv('label-the-sky/csv/dr1.csv')
df = df[(df.photoflag==0)&(df.ndet==12)]

fig, axs = plt.subplots(4, 3, figsize=(10, 10), sharex=True, tight_layout=True)
err = [0.02, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1, 5]
for ix, ax in enumerate(axs.flat):
    e = err[ix]
    x = df.loc[(df.u_err <= e) & (df.f378_err <= e) & (df.f395_err <= e) & (df.f410_err <= e) & (df.f430_err <= e) & (df.g_err <= e) & (df.f515_err <= e) & (df.r_err <= e) & (df.f660_err <= e) & (df.i_err <= e) & (df.f861_err <= e) & (df.z_err <= e), 'r'].values
    size = x.shape[0]
    ax.hist(x)
    ax.set_title('e<={}; {} obj;\n[{}, {}]'.format(e, size, x.min(), x.max()))

plt.savefig('hists.png')
