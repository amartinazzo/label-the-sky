import numpy as np
import os
import sys
from time import time
import yaml

import _base as b
import gen_plots as p
from utils import glob_re

def evaluate(base_dir, timestamp, backbone, n_channels, pretraining_dataset, finetune, split):
    model_name = f'{timestamp}_{backbone}_{n_channels}_{pretraining_dataset}_clf_ft{int(finetune)}'
    weights_file = os.path.join(base_dir, 'trained_models', model_name+'.h5')

    print(weights_file)

    trainer = b.Trainer(
        backbone=backbone,
        n_channels=n_channels,
        output_type='class',
        base_dir=base_dir,
        model_name=model_name,
        weights=weights_file
    )

    trainer.load_weights(weights_file, skip_mismatch=False)

    X_test, y_test = trainer.load_data(dataset='clf', split=split)

    start = time()

    y_hat, X_features = trainer.extract_features_and_predict(X_test)
    np.save(os.path.join(base_dir, 'npy', f'yhat_{split}_{model_name}'), y_hat)
    np.save(os.path.join(base_dir, 'npy', f'Xf_{split}_{model_name}'), X_features)
    trainer.evaluate(X_test, y_test)
    print('--- minutes taken:', int((time() - start) / 60))

def get_dataset_label(filename):
    dataset = filename.split('_')[-3]
    if dataset=='imagenet':
        return 'ImageNet'
    elif dataset=='unlabeled':
        return 'magnitudes'
    return dataset

def get_finetuning_suffix(filename):
    ft = bool(int(filename.split('_ft')[1][0]))
    if ft:
        return 'w/ finetuning'
    return 'w/o finetuning'

def get_nr_channels(filename):
    return ''.join(filter(str.isdigit, filename.split('/')[-1].split('_ft')[0]))[-2:].strip('0') + ' ch'

if __name__ == '__main__':
    b.set_random_seeds()

    if len(sys.argv) < 2:
        print('usage: python {} <yaml_file> <skip_predictions>'.format(sys.argv[0]))
        exit(1)

    config_file = sys.argv[1]
    skip_predictions = bool(int(sys.argv[2])) if sys.argv[2] else True
    config = yaml.load(open(config_file))

    base_dir = os.environ['HOME']
    timestamp = config['timestamp']
    backbone_lst = config['backbones']
    n_channels_lst =config['n_channels']
    pretraining_dataset_lst = config['pretraining_datasets']
    finetune_lst = config['finetune']
    split = config['eval_split']

    if not skip_predictions:
        print('computing feature vectors and predictions')
        for backbone in backbone_lst:
            for n_channels in n_channels_lst:
                for pretraining_dataset in pretraining_dataset_lst:
                    for finetune in finetune_lst:
                        if (not pretraining_dataset and finetune) or (pretraining_dataset=='imagenet' and n_channels!=3):
                            continue
                        evaluate(base_dir, timestamp, backbone, n_channels, pretraining_dataset, finetune, split)

    backbone = backbone_lst[0]

    p.set_plt_style()
    n_plots = 8

    print(f'1/{n_plots} plotting pretraining loss curves')
    file_list = glob_re(os.path.join(base_dir, 'mnt/history'), f'{timestamp}_{backbone}_(12|05|03)_unlabeled.json')
    p.metric_curve(
        file_list=file_list,
        plt_labels=[get_nr_channels(f) for f in file_list],
        output_file='figures/exp_pretraining.pdf',
        metric='val_loss')

    print(f'2/{n_plots} plotting RGB vs imagenet clf')
    file_list = glob_re(os.path.join(base_dir, 'mnt/history'), f'{timestamp}_{backbone}_03_(unlabeled|imagenet)_clf_ft(1|0).json')
    p.metric_curve(
        file_list=file_list,
        plt_labels=[get_dataset_label(f) + ' ' + get_finetuning_suffix(f) for f in file_list],
        output_file='figures/exp_clf_rgb-imagenet.pdf',
        metric='val_accuracy',
        paired=True,
        color_pos=2)

    print(f'3/{n_plots} plotting RGB vs imagenet clf lowdata')
    file_list = glob_re(os.path.join(base_dir, 'mnt/history'), f'{timestamp}_{backbone}_03_(unlabeled|imagenet)_clf_ft(1|0)_lowdata.json')
    p.lowdata_curve(
        file_list=file_list,
        plt_labels=[get_dataset_label(f) + ' ' + get_finetuning_suffix(f) for f in file_list],
        output_file='figures/exp_clf_rgb-imagenet_lowdata.pdf',
        metric='val_accuracy',
        paired=True,
        color_pos=2)

    print(f'4/{n_plots} plotting channels clf')
    file_list = glob_re(os.path.join(base_dir, 'mnt/history'), f'{timestamp}_{backbone}_(12|05|03)_unlabeled_clf_ft(1|0).json')
    p.metric_curve(
        file_list=file_list,
        plt_labels=[get_nr_channels(f) + get_finetuning_suffix(f) for f in file_list],
        output_file='figures/exp_clf_channels.pdf',
        metric='val_accuracy',
        paired=True)

    print(f'5/{n_plots} plotting channels clf lowdata')
    file_list = glob_re(os.path.join(base_dir, 'mnt/history'), f'{timestamp}_{backbone}_(12|05|03)_unlabeled_clf_ft(1|0)_lowdata.json')
    p.lowdata_curve(
        file_list=file_list,
        plt_labels=[get_nr_channels(f) + ' ch ' + get_finetuning_suffix(f) for f in file_list],
        output_file='figures/exp_clf_channels_lowdata.pdf',
        metric='val_accuracy',
        paired=True)

    print(f'6/{n_plots} plotting channels clf accuracy vs r-magnitude')
    file_list = glob_re(os.path.join(base_dir, 'npy'), f'yhat_{split}_{timestamp}_{backbone}_(12|05|03)_unlabeled_clf_ft1.npy')
    p.acc_attribute_curve(
        file_list=file_list,
        plt_labels=[get_nr_channels(f) for f in file_list],
        output_file=f'figures/exp_clf_channels_acc-r_{split}.pdf',
        dataset_file='datasets/clf.csv',
        split=split,
        attribute='r',
        legend_location='lower left')

    print(f'7/{n_plots} plotting umap projections')
    file_list = glob_re(os.path.join(base_dir, 'npy'), f'Xf_{split}_{timestamp}_{backbone}_(12|05|03)_(imagenet|unlabeled)_clf_ft1.npy')
    p.umap_scatter(
        file_list=file_list,
        plt_labels=[get_dataset_label(f) + '; ' + get_nr_channels(f) for f in file_list],
        output_file=f'figures/exp_umap_{split}.pdf')

    print(f'8/{n_plots} plotting umap projections colored by class')
    file_list = glob_re(os.path.join(base_dir, 'npy'), f'Xf_{split}_{timestamp}_{backbone}_(12|05|03)_(imagenet|unlabeled)_clf_ft1.npy')
    for neigh in [50, 100, 200, 500]:
        p.umap_scatter(
            file_list=file_list,
            plt_labels=[get_dataset_label(f) + '; ' + get_nr_channels(f) for f in file_list],
            output_file=f'figures/exp_umap_{split}_classes_neighbours{neigh}.pdf',
            dataset_file='datasets/clf.csv',
            split=split,
            color_attribute='class')
