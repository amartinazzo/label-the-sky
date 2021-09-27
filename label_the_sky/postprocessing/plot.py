from label_the_sky.config import CLASS_MAP
from glob import glob
import json
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc
from MulticoreTSNE import MulticoreTSNE as TSNE
from umap import UMAP


'''
color picker: http://colorschemedesigner.com/csd-3.5
blue, yellow, pink, green
12, 5, 3, imagenet
'''
COLORS_U = ['#1240AB', '#FFAA00', '#CD0074', '#808080']
COLORS = ['#1240AB', '#FFAA00', '#CD0074', '#009100']
COLORS_PAIRED = ['#1240AB', '#6C8CD5', '#FFAA00', '#FFD073', '#CD0074', '#E667AF', '#009100', '#00cc00']
LEGEND_LOCATION = 'lower right'


def set_plt_style():
    plt.style.use('seaborn')
    nice_fonts = {
            'text.usetex': False,#True,
            'font.family': 'serif',
            'axes.labelsize': 8,
            'font.size': 8,
            'legend.fontsize': 8,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
    }
    mpl.rcParams.update(nice_fonts)

def set_size(width='thesis', fraction=1, subplots=[1, 1]):
    '''
    auxiliary func to generate latex-friendly plots
    based on https://jwalton.info/Embed-Publication-Matplotlib-Latex
    '''
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

def clear_plot():
    plt.clf()

def set_colors(ax, paired=False, pos_init=0):
    if paired:
        pallete = COLORS_PAIRED
    else:
        pallete = COLORS
    pos = pos_init*2 if paired else pos_init
    pallete = pallete[pos:]
    ax.set_prop_cycle(color=pallete)

def metric_curve(
    file_list, plt_labels, output_file,
    paired=False, color_pos=0, metric='val_loss', metric_scaling_factor=None, legend_location=LEGEND_LOCATION):
    _, ax = plt.subplots(figsize=set_size(), clear=True)
    set_colors(ax=ax, paired=paired, pos_init=color_pos)

    max_iterations = -1
    n = len(file_list)
    for ix, filename in enumerate(file_list):
        history = json.load(open(filename, 'r'))
        metric_ = [run[f'{metric}'] for run in history]
        metric_ = np.array(metric_)
        if metric_scaling_factor:
            metric_ = metric_ * metric_scaling_factor
        means = metric_.mean(axis=0)
        errors = metric_.std(axis=0, ddof=1)
        iterations = range(means.shape[0])
        if means.shape[0] > max_iterations:
            max_iterations = means.shape[0]
        plt.plot(iterations, means, label=plt_labels[ix], linewidth=1, zorder=n-ix)
        plt.fill_between(iterations, means-errors, means+errors, alpha=0.5)
    plt.legend(loc=legend_location)
    plt.xlim(0, max_iterations)
    plt.xlabel('# of iterations')
    plt.ylabel(metric)
    plt.savefig(output_file, format='pdf', bbox_inches='tight')

def attributes_scatter(
    attribute_x, attribute_y, label_x, label_y, output_file, dataset_file,
    transform_fn=lambda y: y):
    _, ax = plt.subplots(figsize=set_size(), clear=True)
    set_colors(ax=ax)

    df = pd.read_csv(dataset_file)
    if attribute_x not in df.columns or attribute_y not in df.columns:
        raise ValueError('attribute must be one of:', df.columns)

    x = df[attribute_x].values
    y = df[attribute_y].apply(transform_fn).values
    plt.scatter(x, y, alpha=0.5, marker='.', s=4, edgecolors='none')
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.savefig(output_file, format='pdf', bbox_inches='tight')

def score_attribute_scatter(
    yhat_files, plt_labels, output_file, dataset_file, split, attribute,
    legend_location=LEGEND_LOCATION):
    '''
        yhat_files:     list of npy filepaths with y_hat outputs, shape (n_samples, n_classes)
        dataset_file:   path of csv file 
        split:          dataset split from which yhat was computed (train, val, test)
        attribute:      object attribute to plot on x-axis (fwhm, magnitude, magnitude error)
        output_file:    path of output image
    '''
    _, ax = plt.subplots(figsize=set_size(), clear=True)
    set_colors(ax=ax)

    df = pd.read_csv(dataset_file)
    if attribute not in df.columns:
        raise ValueError('attribute must be one of:', df.columns)

    df = df[df.split==split]
    attribute_vals = df[attribute].values
    class_names = df['class'].values
    y = [CLASS_MAP[c] for c in class_names] # ground truth class of each sample

    for ix, yhat_file in enumerate(yhat_files):
        yhat = np.load(yhat_file)
        yhat_scores = yhat[range(yhat.shape[0]), y] # get scores predicted for ground truth classes
        plt.scatter(attribute_vals, yhat_scores, label=plt_labels[ix], alpha=0.2, marker='.', s=4, edgecolors='none')
    plt.xlabel(attribute)
    plt.ylabel('yhat')
    plt.legend(loc=legend_location)
    plt.savefig(output_file, format='pdf', bbox_inches='tight')

def acc_attribute_curve(
    file_list, plt_labels, output_file, dataset_file, split, attribute,
    paired=False, nbins=150, legend_location=LEGEND_LOCATION):
    '''
    plot accuracy vs attribute curve. use bins with approximately uniform number of samples.
    '''
    _, ax = plt.subplots(figsize=set_size(), clear=True)
    set_colors(ax=ax, paired=paired)

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

    area_max = max(attribute_vals) - min(attribute_vals)

    y = y[sorted_idx]

    n = len(file_list)
    for ix, f in enumerate(file_list):
        yhat = np.load(f)
        yhat = np.argmax(yhat, axis=1)
        yhat = yhat[sorted_idx]
        is_correct = yhat==y
        is_correct = np.array_split(is_correct, nbins)
        acc = [sum(subarr) / len(subarr) for subarr in is_correct]
        area = round(auc(attribute_vals, acc)/area_max, 4) # area normalized to (0, 1)
        plt.plot(attribute_vals, acc, label=plt_labels[ix] + '; AUC: ' + str(area), zorder=n-ix, linewidth=1)
    plt.xlabel(attribute)
    plt.ylabel(f'{split} accuracy')
    plt.legend(loc=legend_location)
    plt.savefig(output_file, format='pdf', bbox_inches='tight')

def lowdata_curve(
    file_list, plt_labels, output_file,
    paired=False, color_pos=0, metric='val_accuracy', agg_fn=np.max, size_increment=100, n_subsets=20, legend_location=LEGEND_LOCATION):
    _, ax = plt.subplots(figsize=set_size(), clear=True)
    set_colors(ax=ax, paired=paired, pos_init=color_pos)

    n = len(file_list)
    for ix, f in enumerate(file_list):
        history = json.load(open(f, 'r'))
        metric_means = []
        metric_stds = []
        sample_sizes = [size_increment*n for n in range(1, n_subsets+1)]
        for run_history in history:
            metrics = np.array([agg_fn(run[f'{metric}']) for run in run_history])
            metric_means.append(metrics.mean())
            metric_stds.append(metrics.std(ddof=1))
        metric_means = np.array(metric_means)
        metric_stds = np.array(metric_stds)
        plt.plot(sample_sizes, metric_means, label=plt_labels[ix], linewidth=1, zorder=n-ix)
        plt.fill_between(sample_sizes, metric_means-metric_stds, metric_means+metric_stds, alpha=0.5)
    plt.legend(loc=legend_location)
    plt.xlabel('# of samples')
    plt.ylabel(metric)
    plt.savefig(output_file, format='pdf', bbox_inches='tight')

def hist(dataset_file, output_file, attribute, color_attribute=None, color_pos=0, paired=False):
    _, ax = plt.subplots(figsize=set_size(), clear=True)
    set_colors(ax=ax, paired=paired, pos_init=3)

    df = pd.read_csv(dataset_file)
    sns.kdeplot(df[attribute].values, ax=ax, alpha=0.5, fill=True, label='all', linewidth=1)

    set_colors(ax=ax, paired=paired, pos_init=color_pos)
    if color_attribute:
        colors = sorted(df[color_attribute].unique())
        for c in colors:
            sns.kdeplot(df[df[color_attribute]==c][attribute].values, ax=ax, alpha=0.2, fill=True, label=c.lower(), linewidth=1)
        plt.legend(loc='upper left')
    plt.xlabel(attribute)
    plt.ylabel('density')
    plt.ylim(0, 1)
    plt.savefig(output_file, format='pdf', bbox_inches='tight')

def hist_datasets(dataset_files, output_file, attribute, plt_labels, color_pos=0, paired=False):
    _, ax = plt.subplots(figsize=set_size(), clear=True)
    set_colors(ax=ax, paired=paired, pos_init=color_pos)

    for ix, dataset in enumerate(dataset_files):
        df = pd.read_csv(dataset)
        sns.kdeplot(df[attribute].values, ax=ax, alpha=0.5, fill=True, label=plt_labels[ix], linewidth=1)
    plt.legend(loc='upper right')
    plt.xlabel(attribute)
    plt.ylabel('density')
    plt.ylim(0, 1)
    plt.savefig(output_file, format='pdf', bbox_inches='tight')

def projection_scatter(
    file_list, plt_labels, output_file, algorithm='umap',
    dataset_file=None, split=None, color_attribute=None, n_cols=2, n_neighbors=15):
    if len(file_list) > 1:
        marker_size = 6
        ncols, nrows = n_cols, np.ceil(len(file_list)/n_cols).astype(int)
    else:
        marker_size = 12
        ncols, nrows = 1, 1
    subplots = [nrows, ncols]
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=set_size(subplots=subplots), clear=True)

    if color_attribute:
        df = pd.read_csv(dataset_file)
        df = df[df.split==split]
        y = df[color_attribute].values

    for ix, f in enumerate(file_list):
        X_f = np.load(f)
        if algorithm=='umap':
            X_projected = UMAP(n_neighbors=n_neighbors, random_state=42).fit_transform(X_f)
        elif algorithm=='tsne':
            X_projected = TSNE(n_jobs=multiprocessing.cpu_count(), perplexity=50, n_iter=5000, random_state=42).fit_transform(X_f)
        if len(file_list) > n_cols:
            ax_ = ax[ix//n_cols, ix%n_cols]
        elif len(file_list) == n_cols:
            ax_ = ax[ix]
        else:
            ax_ = ax
        if color_attribute and color_attribute=='class':
            # discretized colors
            y = list(y) + ['UNKNOWN'] * (X_projected.shape[0] - len(y)) # pad
            colors = [COLORS_U[CLASS_MAP[c]] for c in y]
            ax_.scatter(X_projected[:, 0], X_projected[:, 1], alpha=0.2, marker='.', s=marker_size, edgecolors='none', c=colors)
        elif color_attribute:
            # continuous color map
            im = ax_.scatter(X_projected[:, 0], X_projected[:, 1], alpha=0.2, marker='.', s=marker_size, edgecolors='none', c=y, cmap='plasma')
            if ix==len(file_list)-1:
                cbar = fig.colorbar(im, ax=ax_, pad=0.05)
                cbar.ax.tick_params(size=0)
                cbar.set_label(color_attribute)
                cbar.set_alpha(1)
                cbar.draw_all()
        else:
            ax_.scatter(X_projected[:, 0], X_projected[:, 1], alpha=0.2, marker='.', s=marker_size, edgecolors='none', c=COLORS[ix])
        ax_.set_xticks([])
        ax_.set_yticks([])
        ax_.set_xlabel(plt_labels[ix])

    if color_attribute and color_attribute=='class':
        legend_els = [Line2D(
            [0], [0], lw=0, marker='.', markerfacecolor=COLORS_U[ix], markersize=6, label=[*CLASS_MAP.keys()][ix].lower()
            ) for ix in range(len(np.unique(y)))]
        ax_ = ax[0, 0] if len(file_list) > 1 else ax
        ax_.legend(handles=legend_els, loc='upper left')
    plt.savefig(output_file, format='pdf', bbox_inches='tight')

def missclassified_samples(file_list, crops_path, dataset_file, output_file, n=10):
    raise NotImplementedError
