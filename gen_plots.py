'''
auxiliary funcs to generate latex-friendly plots
based on: https://jwalton.info/Embed-Publication-Matplotlib-Latex/

'''

from glob import glob
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_fonts():
    plt.style.use('seaborn')
    nice_fonts = {
            'text.usetex': False,#True,
            'font.family': 'serif',
            'axes.labelsize': 6,
            'font.size': 8,
            'legend.fontsize': 8,
            'xtick.labelsize': 6,
            'ytick.labelsize': 6,
    }
    mpl.rcParams.update(nice_fonts)


def set_size(width='thesis', fraction=1, subplots=[1, 1]):
    if width == 'thesis':
        width_pt = 468.33257
    else:
        width_pt = width

    fig_width_pt = width_pt * fraction
    inches_per_pt = 1 / 72.27 # pt to in conversion
    golden_ratio = (5**.5 - 1) / 2

    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio * subplots[0] / subplots[1]

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def make_magnitude_histograms(df_arr, df_names, mag_col='r'):
    cmap = mpl.cm.get_cmap('rainbow_r', len(df_arr))
    alpha = np.linspace(1, 0.4, len(df_arr))
    fig, ax = plt.subplots(figsize=set_size())
    plt.setp(ax, xticks=[12, 14, 16, 18, 20, 22, 24], yticks=[])
    for i in range(len(df_arr)):
        df = pd.read_csv(df_arr[i])
        mags = df[mag_col].values
        plt.hist(mags, bins=50, alpha=0.5, color=cmap(i), label=df_names[i], linewidth=0.)
    plt.legend()
    plt.savefig('svg/magnitude_hists.svg', format='svg', bbox_inches='tight')


def make_err_histograms(arr, rows=3, cols=4, xmax=0.5):
    cmap = mpl.cm.get_cmap('rainbow_r', 12)
    legend = ['U', 'F378', 'F395', 'F410', 'F430', 'G', 'F515', 'R', 'F660', 'I', 'F861', 'Z']
    fig, ax = plt.subplots(rows, cols, figsize=set_size(subplots=[rows, cols]))
    plt.setp(ax, xticks=[0.1, 0.2, 0.3, 0.4, 0.5], yticks=[10000, 20000, 30000, 40000, 50000, 60000, 70000], xticklabels=[], yticklabels=[])
    n = 0
    for i in range(rows):
        for j in range(cols):
            a = arr[(arr[:, n] <= xmax), n]
            ax[i, j].hist(a, bins=25, color=cmap(12-n), edgecolor=cmap(12-n), linewidth=0.0)
            ax[i, j].set_xlabel(legend[n], labelpad=-5)
            ax[i, j].set_xlim(0, xmax)
            n = n+1
    plt.savefig('svg/magnitude_errors.svg', format='svg', bbox_inches='tight')


def make_err_histograms_overlapped(arr, xmax=0.5):
    cmap = mpl.cm.get_cmap('rainbow_r', 12)
    legend = ['U', 'F378', 'F395', 'F410', 'F430', 'G', 'F515', 'R', 'F660', 'I', 'F861', 'Z']
    legend.reverse()
    fig, ax = plt.subplots(figsize=set_size())
    plt.setp(ax, xticks=[0, 0.25, 0.5], yticks=[]) #xticklabels=[])
    for n in range(11, -1, -1):
        a = arr[(arr[:,n]<=xmax),n]
        plt.hist(a, bins=50, color=cmap(n), alpha=0.5, label=legend[n])
        # ax[i,j].set_xlabel(legend[n])
        # ax[i,j].set_xlim(0, xmax)
        # ax[i,j].set_ylim(0, 50000)
    plt.legend()
    plt.xticks([], [])
    plt.savefig('svg/magnitude_errors_overlap.svg', format='svg', bbox_inches='tight')


def make_history_curves(glob_pattern, output_file, metric='loss', color_duos=True):
    files = glob(glob_pattern)
    files.sort()
    print('nr of files', len(files))

    len_cmap = len(files)/2 if color_duos else len(files)
    cmap = mpl.cm.get_cmap('rainbow_r', len_cmap)

    # ax[0] -> backbone loss
    # ax[1] -> top classifier accuracy

    plt.subplots(figsize=set_size())
    for i, f in enumerate(files):
        history = json.load(open(f, 'r'))[0]
        label = '_'.join(f.split('/')[-1].split('_')[1:])
        ft = label[-1]
        marker = 'x' if ft == '0' else 'None'
        color_idx = int(i/2) if color_duos else i
        print(label, '\t', ft, '\t', marker)
        loss = history[f'{metric}']
        loss_val = history[f'val_{metric}']
        plt.plot(range(len(loss)), loss, linewidth=1, color=cmap(color_idx), linestyle='dotted')
        plt.plot(range(len(loss_val)), loss_val, linewidth=1, alpha=0.9, color=cmap(color_idx), markersize=3, markevery=5, marker=marker, label=label)
        # ax[0].plot(range(len(loss)), loss, alpha=0.9, color=cmap(i), label=label)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(output_file, format='svg', bbox_inches='tight')


def gen_scatterplot(x, y, x_label, y_label, output_file):
    # use to generate score vs r-magnitude plots
    plt.plot(x, y, alpha=0.9)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(output_file, format='svg', bbox_inches='tight')
    pass


if __name__ == '__main__':
    set_fonts()

    # make magnitude uncertainty histograms
    # df = pd.read_csv('datasets/clf.csv')
    # df = df.loc[:, [
    #       'u_err','f378_err','f395_err','f410_err','f430_err','g_err',
    #       'f515_err','r_err','f660_err','i_err','f861_err','z_err']]
    # arr = df.values
    # make_err_histograms_overlapped(arr)

    # make training history curves
    # make_history_curves(200220)

    datasets = ['datasets/unlabeled-1-100k.csv', 'datasets/unlabeled-05-100k.csv', 'datasets/unlabeled-01-100k.csv', 'datasets/unlabeled-005-100k.csv']
    dataset_names = ['unlabeled-e1', 'unlabeled-e0.5', 'unlabeled-e0.1', 'unlabeled-e0.05']
    make_magnitude_histograms(datasets, dataset_names)

    # acc_imagenet = [0.500485908649174, 0.9212827988338192, 0.9310009718172984, 0.49951409135082603, 0.500485908649174, 0.49951409135082603, 0.924198250728863, 0.9300291545189504, 0.9203109815354713, 0.8950437317784257, 0.9164237123420796, 0.500485908649174, 0.49951409135082603, 0.8765792031098154, 0.9251700680272109, 0.49951409135082603, 0.49951409135082603, 0.49951409135082603, 0.49951409135082603, 0.49951409135082603]
    # acc = [0.5422740524781341, 0.6491739552964043, 0.652089407191448, 0.6472303206997084, 0.652089407191448, 0.6511175898931001, 0.7366375121477162, 0.6579203109815355, 0.6647230320699709, 0.6482021379980564, 0.6793002915451894, 0.771622934888241, 0.6715257531584062, 0.7774538386783285, 0.6559766763848397, 0.6452866861030127, 0.6559766763848397, 0.7998056365403304, 0.760932944606414, 0.6987366375121478]
    # xlabel = np.linspace(0.5,10, 20)
    # plt.plot(xlabel, acc_imagenet, label='ImageNet features')
    # plt.plot(xlabel, acc, label='magnitude regressor features')
    # plt.legend()
    # plt.xlabel('nr of training examples (thousands)')
    # plt.ylabel('accuracy')
    # plt.savefig(f'accuracies.svg', format='svg', bbox_inches='tight')