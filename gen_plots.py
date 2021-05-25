'''
auxiliary funcs to generate latex-friendly plots
based on: https://jwalton.info/Embed-Publication-Matplotlib-Latex/

'''

from constants import CLASS_MAP
from glob import glob
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


LEGEND_LOCATION = 'upper right'


def set_plt_style():
    plt.style.use('seaborn')
    nice_fonts = {
            'text.usetex': False,#True,
            'font.family': 'serif',
            'axes.labelsize': 6,
            'font.size': 8,
            'legend.fontsize': 6,
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


def make_magnitude_histograms(df_arr, df_names, output_file, mag_col='r'):
    cmap = mpl.cm.get_cmap('rainbow_r', len(df_arr))
    alpha = np.linspace(1, 0.4, len(df_arr))
    fig, ax = plt.subplots(figsize=set_size())
    plt.setp(ax, xticks=[12, 14, 16, 18, 20, 22, 24], yticks=[])
    for i in range(len(df_arr)):
        df = pd.read_csv(df_arr[i])
        mags = df[mag_col].values
        plt.hist(mags, bins=50, alpha=0.5, color=cmap(i), label=df_names[i], linewidth=0.)
    plt.legend()
    plt.savefig(output_file, format='pdf', bbox_inches='tight')


def make_err_histograms(arr, output_file, rows=3, cols=4, xmax=0.5):
    cmap = mpl.cm.get_cmap('rainbow_r', 12)
    legend = ['U', 'F378', 'F395', 'F410', 'F430', 'G', 'F515', 'R', 'F660', 'I', 'F861', 'Z']
    fig, ax = plt.subplots(rows, cols, figsize=set_size(subplots=[rows, cols]))
    plt.setp(
        ax,
        xticks=[0.1, 0.2, 0.3, 0.4, 0.5],
        yticks=[10000, 20000, 30000, 40000, 50000, 60000, 70000],
        xticklabels=[0.1, 0.2, 0.3, 0.4, 0.5],
        yticklabels=[10000, None, None, None, 50000, None, None]
        )
    n = 0
    for i in range(rows):
        for j in range(cols):
            a = arr[(arr[:, n] <= xmax), n]
            ax[i, j].hist(a, bins=25, color=cmap(12-n), edgecolor=cmap(12-n), linewidth=0.0)
            ax[i, j].set_xlabel(legend[n], labelpad=-5)
            ax[i, j].set_xlim(0, xmax)
            n = n+1
    plt.savefig(output_file, format='pdf', bbox_inches='tight')


def make_err_histograms_overlapped(arr, output_file, xmax=0.5):
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
    plt.savefig(output_file, format='pdf', bbox_inches='tight')


def make_trainval_curves(glob_pattern, output_file, metric='loss', color_duos=True):
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
    plt.savefig(output_file, format='pdf', bbox_inches='tight')


def make_metrics_curves(glob_pattern, output_file, metrics=['val_loss'], legend_location=LEGEND_LOCATION):
    if type(glob_pattern) != list:
        pattern_lst = [glob_pattern]
    else:
        pattern_lst = glob_pattern

    files = []
    for pattern in pattern_lst:
        files = files + glob(pattern)
    files.sort()
    print('nr of files', len(files), metrics)

    max_iterations = -1
    _, ax = plt.subplots(figsize=set_size())
    set_colors(ax=ax, n_colors=len(files))

    for f in files:
        history = json.load(open(f, 'r'))
        splt = f.split('_')
        if len(splt)>4:
            plt_label = f'{splt[3]}; {splt[2]} channels; {splt[5][2]}'
        else:
            plt_label = f'{splt[2]} channels'

        for m in metrics:
            metric = [run[f'{m}'] for run in history]
            metric = np.array(metric)
            means = metric.mean(axis=0)
            errors = metric.std(axis=0, ddof=1)
            iterations = range(means.shape[0])
            if means.shape[0] > max_iterations:
                max_iterations = means.shape[0]
            plt.plot(iterations, means, linewidth=1, label=plt_label)
            plt.fill_between(iterations, means-errors, means+errors, alpha=0.5)
    legend_ncols = len(files) // 3
    plt.legend(loc=legend_location, ncol=legend_ncols)
    plt.xlim(0, max_iterations)
    plt.xlabel('# of iterations')
    plt.ylabel(metrics[0])
    plt.savefig(output_file, format='pdf', bbox_inches='tight')


def gen_score_vs_attribute_plot(yhat_files_glob, dataset_file, split, attribute, output_file):
    '''
        yhat_files_glob:glob pattern for paths of npy files with y_hat outputs, shape (n_samples, n_classes)
        dataset_file:   path of csv file 
        split:          dataset split from which yhat was computed (train, val, test)
        attribute:      object attribute to plot on x-axis (fwhm, magnitude, magnitude error)
        output_file:    path of output image
    '''
    df = pd.read_csv(dataset_file)
    if attribute not in df.columns:
        raise ValueError('attribute must be one of:', df.columns)

    df = df[df.split==split]
    attribute_vals = df[attribute].values
    class_names = df['class'].values
    y = [CLASS_MAP[c] for c in class_names] # ground truth class of each sample

    yhat_files = glob(yhat_files_glob)
    yhat_files.sort()
    print(yhat_files)

    for ix, yhat_file in enumerate(yhat_files):
        yhat = np.load(yhat_file)
        yhat_scores = yhat[range(yhat.shape[0]), y] # get scores predicted for ground truth classes
        plt_label = yhat_file.split('_')[4] + ' channels'
        print(attribute_vals.shape, yhat.shape, yhat_scores.shape)
        plt.scatter(attribute_vals, np.log(yhat_scores), c=COLORS[ix], label=plt_label, alpha=0.2, s=2)
    plt.xlabel(attribute)
    plt.ylabel('yhat')
    plt.legend(loc='lower left')
    plt.savefig(output_file, format='pdf', bbox_inches='tight')


def acc_attribute_bins(yhat_files_glob, dataset_file, split, attribute, output_file, nbins=100):
    '''
    plot accuracy vs attribute curve. use bins with approximate number of samples.
    '''
    df = pd.read_csv(dataset_file)
    if attribute not in df.columns:
        raise ValueError('attribute must be one of:', df.columns)

    df = df[df.split==split]
    attribute_vals = df[attribute].values
    class_names = df['class'].values
    y = np.array([CLASS_MAP[c] for c in class_names]) # ground truth class of each sample

    sorted_idx = np.argsort(attribute_vals)
    attribute_vals = attribute_vals[sorted_idx]
    attribute_vals = np.array_split(attribute_vals, nbins)
    attribute_vals = [np.median(subarr) for subarr in attribute_vals]

    y = y[sorted_idx]

    yhat_files = glob(yhat_files_glob)
    yhat_files.sort()
    print(yhat_files)

    fig, ax = plt.subplots(figsize=set_size())
    set_colors(ax=ax, n_colors=len(yhat_files))

    for ix, yhat_file in enumerate(yhat_files):
        plt_label = yhat_file.split('_')[4] + ' channels'
        yhat = np.load(yhat_file)
        yhat = np.argmax(yhat, axis=1)
        yhat = yhat[sorted_idx]
        is_correct = yhat==y
        is_correct = np.array_split(is_correct, nbins)
        acc = [sum(subarr) / len(subarr) for subarr in is_correct]
        plt.plot(attribute_vals, acc, label=plt_label)
    plt.xlabel(attribute)
    plt.ylabel('acc')
    plt.legend(loc='lower left')
    plt.savefig(output_file, format='pdf', bbox_inches='tight')

def clear_plot():
    plt.clf()

def set_colors(ax, n_colors):
    ax.set_prop_cycle(color=sns.color_palette('husl', n_colors=n_colors))

def plot_lowdata(glob_pattern, output_file, metric='val_accuracy', agg_fn=np.max, size_increment=100, n_subsets=20, legend_location=LEGEND_LOCATION):
    if type(glob_pattern) != list:
        pattern_lst = [glob_pattern]
    else:
        pattern_lst = glob_pattern

    files = []
    for pattern in pattern_lst:
        files_tmp = glob(pattern)
        files = files + files_tmp
    files.sort()
    print('nr of files', len(files))

    fig, ax = plt.subplots(figsize=set_size())
    set_colors(ax=ax, n_colors=len(files))

    for f in files:
        history = json.load(open(f, 'r'))
        splt = f.split('_')
        plt_label = f'{splt[3]}; {splt[2]} channels; {splt[5]}'

        metric_means = []
        metric_stds = []
        sample_sizes = [size_increment*n for n in range(1, n_subsets+1)]
        for run_history in history:
            metrics = np.array([agg_fn(run[f'{metric}']) for run in run_history])
            metric_means.append(metrics.mean())
            metric_stds.append(metrics.std(ddof=1))
        metric_means = np.array(metric_means)
        metric_stds = np.array(metric_stds)
        plt.plot(sample_sizes, metric_means, label=plt_label, linewidth=1)
        plt.fill_between(sample_sizes, metric_means-metric_stds, metric_means+metric_stds, alpha=0.5)
        # ax.annotate(plt_label, xy = (sample_sizes[-1]-200, metric_means[-1]), xytext = (sample_sizes[-1]-200, metric_means[-1]))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('# of samples')
    plt.ylabel(metric)
    plt.savefig(output_file, format='pdf', bbox_inches='tight')

def make_error_analysis():
    raise NotImplementedError

def gen_2d_projection(X_features_path, dataset_path):
    raise NotImplementedError
