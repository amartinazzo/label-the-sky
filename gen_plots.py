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
from _datagen import CLASS_MAP


def set_plt_style():
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


def make_err_histograms_overlapped(arr, xmax=0.5, output_file):
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


def make_metrics_curves(glob_pattern, output_file, metrics=['val_loss'], legend_location='upper right'):
    if type(glob_pattern) != list:
        pattern_lst = [glob_pattern]
    else:
        pattern_lst = glob_pattern

    markers = [None, '$x$']
    max_iterations = -1
    fig, ax = plt.subplots(figsize=set_size())

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for ix, pattern in enumerate(pattern_lst):
        files = glob(pattern)
        files.sort()
        print('nr of files', len(files))
        print('metrics', metrics)

        for ix_f, f in enumerate(files):
            history = json.load(open(f, 'r'))
            n_runs = len(history)
            plt_label = f.split('_')[2] + ' channels'

            for m in metrics:
                metric = [history[n][f'{m}'] for n in range(n_runs)]
                metric = np.array(metric)
                means = metric.mean(axis=0)
                errors = metric.std(axis=0, ddof=1)
                iterations = range(means.shape[0])
                if means.shape[0] > max_iterations:
                    max_iterations = means.shape[0]
                plt.plot(iterations, means, color=colors[ix_f], linewidth=1, label=plt_label, marker=markers[ix], markevery=10)
                plt.fill_between(iterations, means-errors, means+errors, color=colors[ix_f], alpha=0.5)
        if ix==0:
            plt.legend(loc=legend_location)
    plt.xlim(0, max_iterations)
    plt.xlabel('# of iterations')
    plt.ylabel(metrics[0])
    plt.savefig(output_file, format='pdf', bbox_inches='tight')


def gen_score_vs_attribute_plot(yhat_path, dataset_path, split, attribute, include, output_file):
    '''
        yhat_path:      path of npy file with y_hat outputs, shape (n_samples, n_classes)
        dataset_path:   path of csv file 
        split:          dataset split from which yhat was computed (train, val, test)
        attribute:      object attribute to plot on x-axis (fwhm, magnitude, magnitude error)
        include:        which objects from yhat to include on the plot
                        'all'       (plot all samples)
                        'correct'   (plot correctly classified samples)
                        'incorrect' (plot incorrecly classifier samples)
        output_file:    path of output image
    '''
    INCLUDE_VALS = ['all', 'correct', 'incorrect']
    if include not in INCLUDE_VALS:
        raise ValueError('include must be one of:', INCLUDE_VALS)

    df = pd.read_csv(dataset_path)
    if attribute not in df.columns:
        raise ValueError('attribute must be one of:', df.columns)

    df = df[df.split==split]
    attribute_vals = df[attribute].values
    class_names = df['class'].values
    y = [CLASS_MAP[c] for c in class_names]

    yhat = np.load(yhat_path)
    yhat_class = np.argmax(yhat, axis=1)
    yhat_score = yhat[yhat_class]

    if include != 'all':
        include_idx = y == yhat_score if include == 'correct' else y != yhat_score
        attribute_vals = attribute_vals[include_idx]
        class_names = class_names[include_idx]
        yhat_class = yhat_class[include_idx]
        yhat_score = yhat_score[include_idx]

    gen_scatterplot(
        x=attribute_vals,
        y=yhat_score,
        colors=yhat_class,
        labels=class_names,
        x_label=attribute,
        y_label='score',
        output_file=output_file
    )


def gen_scatterplot(x, y, colors, labels, x_label, y_label, output_file):
    plt.plot(x, y, c=colors, labels=labels, alpha=0.5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(output_file, format='pdf', bbox_inches='tight')


def gen_2d_projection(X_features_path, dataset_path):
    # TODO implement
    raise NotImplementedError
