# post-processing pipeline


import matplotlib.cm as cm
from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerPathCollection
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sklearn.metrics as metrics
import sys
from visualize import update_legend_marker


classes = ['GALAXY', 'STAR', 'QSO']
n_classes = len(classes)

colors = [cm.Dark2(x) for x in range(2*n_classes)]


def gen_accuracy_mag_plot(filepath, df, y_true, y_pred, mag_min=16, mag_max=25, bins=4):
    df = df[(df.ndet==12)&(df.photoflag==0)&(df.split=='val')].reset_index()
    intervals = np.linspace(mag_min, mag_max, bins*(mag_max-mag_min)+1)
    mags = []
    acc = [[], [], []]

    yp_arg = np.argmax(y_pred, axis=1)
    yt_arg = np.argmax(y_true, axis=1)

    plt.clf()
    for ix in range(len(intervals)-1):
        idx = df.r.between(intervals[ix], intervals[ix+1])
        yt = yt_arg[idx]
        yp = yp_arg[idx]
        if len(np.unique(yt)) != n_classes:
            print('skipping', intervals[ix])
            continue
        cm = metrics.confusion_matrix(yt, yp)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        if cm.shape[0]==n_classes:
            mags.append(intervals[ix])
            for c in range(n_classes):
                acc[c].append(cm[c,c])

    for c in range(n_classes):
        plt.plot(mags, acc[c], c=colors[c])
    plt.legend(classes)
    plt.ylabel('ACCURACY')
    plt.xlabel('MAGNITUDE (R)')
    plt.tight_layout()
    plt.savefig(f'{filepath}.png')



def gen_roc_curve(filepath, y_true, y_pred):
    plt.clf()
    legend = []
    for c in range(n_classes):
        yt = y_true[:,c]
        yp = y_pred[:,c]
        fp, tp, thresholds = metrics.roc_curve(yt, yp)
        auc = np.round(metrics.auc(fp, tp),4)
        plt.plot(fp, tp, c=colors[c])
        legend.append(classes[c] + ', AUC = ' + str(auc))
    plt.legend(legend)
    plt.xlabel('FP')
    plt.ylabel('TP')
    plt.gca().set_ylim(0, 1.1)
    plt.savefig(f'{filepath}.png')


def gen_projection(filepath, X, y):
    plt.clf()
    y_arg = np.argmax(y, axis=1)
    for c in range(n_classes):
        idx = y_arg==c
        xc = X[idx]
        print(len(xc))
        plt.scatter(xc[:,0], xc[:,1], c=[colors[c]], s=4, marker='.', alpha=0.2, label=classes[c])
    plt.legend(handler_map={PathCollection : HandlerPathCollection(update_func=update_legend_marker)})
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.tight_layout()
    plt.savefig(f'{filepath}.png')


if __name__ == '__main__':

    if len(sys.argv) != 7:
        print('usage: python %s <results_dir> <csv_file> <target> <nbands> <feature_vector_dim> <timestamp>' % sys.argv[0])
        exit(1)

    results_dir = sys.argv[-6]
    csv_file = sys.argv[-5]
    target = sys.argv[-4]
    n_bands = int(sys.argv[-3])
    output_dim = int(sys.argv[-2])
    timestamp = 191019 #sys.argv[-1]

    model_name = '{}_{}_{}bands_{}'.format(timestamp, target, n_bands, output_dim)
    print(model_name)
    if model_name == '191019_classes_12bands_1024':
        model_name = '191018_classes_12bands_1024'

    print('\n\nresults_dir', results_dir)
    print('target', target)
    print('n_bands', n_bands)
    print('output_dim', output_dim)

    y_train = np.load(os.path.join(results_dir, f'{model_name}_y_train.npy'))
    y_val = np.load(os.path.join(results_dir, f'{model_name}_y_val.npy'))
    y_val_hat = np.load(os.path.join(results_dir, f'{model_name}_y_val_hat.npy'))

    # df = pd.read_csv(csv_file)
    # gen_accuracy_mag_plot(f'{model_name}_acc-mag', df, y_val, y_val_hat)
    # gen_roc_curve(f'{model_name}_roc', y_val, y_val_hat)

    val_len = len(y_val)
    X_umap = np.load(os.path.join(results_dir, f'{model_name}_X_features_umap.npy'))
    X_umap = X_umap[-val_len:]
    # y_umap = np.concatenate([y_train, y_val])

    gen_projection(f'{model_name}_umap', X_umap, y_val)