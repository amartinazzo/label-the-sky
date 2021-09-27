import numpy as np
import os
import sys
from time import time

from label_the_sky.config import config
from label_the_sky.training.trainer import Trainer, set_random_seeds
from label_the_sky.training.trainer import MAG_MAX
from label_the_sky.postprocessing import plot as p
from label_the_sky.utils import glob_re, get_dataset_label, get_channels_label, get_finetuning_suffix


def predict_unlabeled(base_dir, timestamp, backbone, n_channels, pretraining_dataset, split, dataset='unlabeled'):
    model_name = f'{timestamp}_{backbone}_{n_channels}_{pretraining_dataset}'
    weights_file = 'imagenet' if pretraining_dataset=='imagenet' else os.path.join(base_dir, 'trained_models', model_name+'.h5')
    print(weights_file)

    trainer = Trainer(
        backbone=backbone,
        n_channels=n_channels,
        output_type='magnitudes',
        base_dir=base_dir,
        model_name=model_name,
        weights=weights_file
    )

    X = trainer.load_data(dataset=dataset, split=split, return_y=False)
    y_hat, X_features = trainer.extract_features_and_predict(X)
    
    suffix = 'u' if dataset=='unlabeled' else ''
    output_name = f'{split}_{timestamp}_{backbone}_{str(n_channels).zfill(2)}_{pretraining_dataset}'
    np.save(os.path.join(base_dir, 'npy', f'y{suffix}hat_{output_name}.npy'), y_hat)
    np.save(os.path.join(base_dir, 'npy', f'X{suffix}f_{output_name}.npy'), X_features)

def predict_clf(base_dir, timestamp, backbone, n_channels, pretraining_dataset, finetune, split, dataset='clf'):
    model_name = f'{timestamp}_{backbone}_{n_channels}_{pretraining_dataset}_clf_ft{int(finetune)}'
    weights_file = os.path.join(base_dir, 'trained_models', model_name+'.h5')
    print(weights_file)

    trainer = Trainer(
        backbone=backbone,
        n_channels=n_channels,
        output_type='class',
        base_dir=base_dir,
        model_name=model_name,
        weights=weights_file
    )
    X, y = trainer.load_data(dataset=dataset, split=split)
    y_hat, X_features = trainer.extract_features_and_predict(X)
    output_name = f'{split}_{timestamp}_{backbone}_{str(n_channels).zfill(2)}_{pretraining_dataset}_clf_ft{int(finetune)}'

    suffix = 'u' if dataset=='unlabeled' else ''
    np.save(os.path.join(base_dir, 'npy', f'y{suffix}hat_{output_name}.npy'), y_hat)
    np.save(os.path.join(base_dir, 'npy', f'X{suffix}f_{output_name}.npy'), X_features)
    # trainer.evaluate(X, y)

def concat_vectors(base_dir, timestamp, backbone, n_channels, pretraining_dataset, finetune, split):
    output_name = f'{split}_{timestamp}_{backbone}_{str(n_channels).zfill(2)}_{pretraining_dataset}'#_clf_ft{int(finetune)}'
    X = np.load(os.path.join(base_dir, 'npy', f'Xf_{output_name}.npy'))
    Xu = np.load(os.path.join(base_dir, 'npy', f'Xuf_{output_name}.npy'))

    X_concat = np.concatenate((X, Xu))
    np.save(os.path.join(base_dir, 'npy', f'Xf-Xuf_{output_name}.npy'), X_concat)


if __name__ == '__main__':
    set_random_seeds()
    skip_predictions = bool(int(sys.argv[1])) if len(sys.argv)>1 else True

    base_dir = os.environ['HOME']
    data_dir = os.environ['DATA_PATH']

    timestamp = config['timestamp']
    backbone_lst = config['backbones']
    n_channels_lst =config['n_channels']
    pretraining_dataset_lst = config['pretraining_datasets']
    finetune_lst = config['finetune']
    split = config['eval_split']
    projection_algo = config['projection_algorithm']
    umap__n_neighbors = config['umap']['n_neighbors']

    if not skip_predictions:
        print('computing feature vectors and predictions')
        for backbone in backbone_lst:
            for n_channels in n_channels_lst:
                for pretraining_dataset in pretraining_dataset_lst:
                    for finetune in finetune_lst:
                        if (not pretraining_dataset and finetune) or (pretraining_dataset=='imagenet' and n_channels!=3):
                            continue
                        predict_unlabeled(base_dir, timestamp, backbone, n_channels, pretraining_dataset, split)
                        for dataset in ['unlabeled', 'clf']:
                            predict_unlabeled(base_dir, timestamp, backbone, n_channels, pretraining_dataset, split)
                        concat_vectors(base_dir, timestamp, backbone, n_channels, pretraining_dataset, finetune, split)

    backbone = backbone_lst[0]

    p.set_plt_style()
    cnt_iterator = iter(range(100))

    try:
        print(f'{str(next(cnt_iterator)).zfill(2)} plotting nr of missing values vs r-magnitude, dr1')
        p.attributes_scatter(
            attribute_x='r_auto', label_x='r',
            attribute_y='nDet_auto', label_y='# missing magnitudes',
            transform_fn=lambda y: 12 - y,
            output_file=f'figures/exp_missingdata-r_dr1.pdf',
            dataset_file=os.path.join(data_dir, 'dr1/dr1_master.csv'))
    except FileNotFoundError as e:
        print(e)

    print(f'{str(next(cnt_iterator)).zfill(2)} plotting r-magnitude distributions, unlabeled')
    p.hist(
        output_file=f'figures/exp_dist_r_unlabeled.pdf',
        dataset_file='datasets/unlabeled.csv',
        attribute='r',
        color_pos=3)

    print(f'{str(next(cnt_iterator)).zfill(2)} plotting r-magnitude distributions, clf')
    p.hist(
        output_file=f'figures/exp_dist_r_clf.pdf',
        dataset_file='datasets/clf.csv',
        attribute='r',
        color_attribute='class')

    print(f'{str(next(cnt_iterator)).zfill(2)} plotting fwhm distributions, unlabeled vs clf')
    file_list = glob_re(os.path.join(base_dir, 'mnt/label-the-sky/datasets'), f'(unlabeled|clf).csv')
    p.hist_datasets(
        output_file=f'figures/exp_dist_fwhm.pdf',
        dataset_files=file_list,
        plt_labels=['$X_{u}$', '$X$'],
        attribute='fwhm')

    print(f'{str(next(cnt_iterator)).zfill(2)} plotting pretraining loss curves')
    file_list = glob_re(os.path.join(base_dir, 'mnt/history'), f'{timestamp}_{backbone}_(12|05|03)_unlabeled.json')
    p.metric_curve(
        file_list=file_list,
        plt_labels=[get_channels_label(f) for f in file_list],
        output_file='figures/exp_pretraining_loss.pdf',
        metric='val_loss',
        metric_scaling_factor=MAG_MAX,
        legend_location='upper right')

    print(f'{str(next(cnt_iterator)).zfill(2)} plotting pretraining times curves')
    file_list = glob_re(os.path.join(base_dir, 'mnt/history'), f'{timestamp}_{backbone}_(12|05|03)_unlabeled.json')
    p.metric_curve(
        file_list=file_list,
        plt_labels=[get_channels_label(f) for f in file_list],
        output_file='figures/exp_pretraining_times.pdf',
        metric='times',
        legend_location='upper right')

    print(f'{str(next(cnt_iterator)).zfill(2)} plotting RGB vs imagenet clf')
    file_list = glob_re(os.path.join(base_dir, 'mnt/history'), f'{timestamp}_{backbone}_03_(unlabeled|imagenet)_clf_ft(1|0).json')
    p.metric_curve(
        file_list=file_list,
        plt_labels=[get_dataset_label(f) + ' ' + get_finetuning_suffix(f) for f in file_list],
        output_file='figures/exp_clf_rgb-imagenet.pdf',
        metric='val_accuracy',
        paired=True,
        color_pos=2)

    print(f'{str(next(cnt_iterator)).zfill(2)} plotting RGB vs imagenet clf lowdata')
    file_list = glob_re(os.path.join(base_dir, 'mnt/history'), f'{timestamp}_{backbone}_03_(unlabeled|imagenet)_clf_ft(1|0)_lowdata.json')
    p.lowdata_curve(
        file_list=file_list,
        plt_labels=[get_dataset_label(f) + ' ' + get_finetuning_suffix(f) for f in file_list],
        output_file='figures/exp_clf_rgb-imagenet_lowdata.pdf',
        metric='val_accuracy',
        paired=True,
        color_pos=2)

    print(f'{str(next(cnt_iterator)).zfill(2)} plotting channels clf')
    file_list = glob_re(os.path.join(base_dir, 'mnt/history'), f'{timestamp}_{backbone}_(12|05|03)_unlabeled_clf_ft(1|0).json')
    p.metric_curve(
        file_list=file_list,
        plt_labels=[get_channels_label(f) + ' ' + get_finetuning_suffix(f) for f in file_list],
        output_file='figures/exp_clf_channels.pdf',
        metric='val_accuracy',
        paired=True)

    print(f'{str(next(cnt_iterator)).zfill(2)} plotting channels clf lowdata')
    file_list = glob_re(os.path.join(base_dir, 'mnt/history'), f'{timestamp}_{backbone}_(12|05|03)_unlabeled_clf_ft(1|0)_lowdata.json')
    p.lowdata_curve(
        file_list=file_list,
        plt_labels=[get_channels_label(f) + ' ' + get_finetuning_suffix(f) for f in file_list],
        output_file='figures/exp_clf_channels_lowdata.pdf',
        metric='val_accuracy',
        paired=True)

    print(f'{str(next(cnt_iterator)).zfill(2)} plotting channels ft vs scratch')
    file_list = glob_re(os.path.join(base_dir, 'mnt/history'), f'{timestamp}_{backbone}_(12|05|03)_(unlabeled_clf_ft1|None_clf_ft0).json')
    p.metric_curve(
        file_list=file_list,
        plt_labels=[get_channels_label(f) + '; ' + get_dataset_label(f) for f in file_list],
        output_file='figures/exp_clf_channels_scratch.pdf',
        metric='val_accuracy',
        paired=True)

    print(f'{str(next(cnt_iterator)).zfill(2)} plotting channels clf ft vs scratch lowdata')
    file_list = glob_re(os.path.join(base_dir, 'mnt/history'), f'{timestamp}_{backbone}_(12|05|03)_(unlabeled_clf_ft1|None_clf_ft0)_lowdata.json')
    p.lowdata_curve(
        file_list=file_list,
        plt_labels=[get_channels_label(f) + '; ' + get_dataset_label(f) for f in file_list],
        output_file='figures/exp_clf_channels_scratch_lowdata.pdf',
        metric='val_accuracy',
        paired=True)

    print(f'{str(next(cnt_iterator)).zfill(2)} plotting channels clf accuracy vs r-magnitude')
    file_list = glob_re(os.path.join(base_dir, 'npy'), f'yhat_{split}_{timestamp}_{backbone}_(12|05|03)_unlabeled_clf_ft1.npy')
    p.acc_attribute_curve(
        file_list=file_list,
        plt_labels=[get_channels_label(f) for f in file_list],
        output_file=f'figures/exp_clf_channels_acc-r_{split}.pdf',
        dataset_file='datasets/clf.csv',
        split=split,
        attribute='r',
        legend_location='lower left')

    print(f'{str(next(cnt_iterator)).zfill(2)} plotting channels clf accuracy vs r-magnitude error')
    file_list = glob_re(os.path.join(base_dir, 'npy'), f'yhat_{split}_{timestamp}_{backbone}_(12|05|03)_unlabeled_clf_ft1.npy')
    p.acc_attribute_curve(
        file_list=file_list,
        plt_labels=[get_channels_label(f) for f in file_list],
        output_file=f'figures/exp_clf_channels_acc-r-err_{split}.pdf',
        dataset_file='datasets/clf.csv',
        split=split,
        attribute='r',
        legend_location='lower left')

    print(f'{str(next(cnt_iterator)).zfill(2)} plotting channels clf accuracy vs fwhm')
    file_list = glob_re(os.path.join(base_dir, 'npy'), f'yhat_{split}_{timestamp}_{backbone}_(12|05|03)_unlabeled_clf_ft1.npy')
    p.acc_attribute_curve(
        file_list=file_list,
        plt_labels=[get_channels_label(f) for f in file_list],
        output_file=f'figures/exp_clf_channels_acc-fwhm_{split}.pdf',
        dataset_file='datasets/clf.csv',
        split=split,
        attribute='fwhm',
        legend_location='lower right')

    print(f'{str(next(cnt_iterator)).zfill(2)} plotting projections colored by r-magnitude, from pretext model features')
    file_list = glob_re(os.path.join(base_dir, 'npy'), f'Xf_unlabeled-{split}_{timestamp}_{backbone}_(12|05|03)_unlabeled.npy')
    p.projection_scatter(
        file_list=file_list,
        plt_labels=[get_dataset_label(f) + '; ' + get_channels_label(f) for f in file_list],
        output_file=f'figures/exp_{projection_algo}_{split}_pretraining_magnitudes.pdf',
        dataset_file='datasets/unlabeled.csv',
        algorithm=projection_algo,
        split=split,
        color_attribute='r',
        n_cols=3,
        n_neighbors=umap__n_neighbors)

    print(f'{str(next(cnt_iterator)).zfill(2)} plotting projections colored by class, from clf features')
    file_list = glob_re(os.path.join(base_dir, 'npy'), f'Xf_{split}_{timestamp}_{backbone}_(12|05|03)_(imagenet|unlabeled)_clf_ft1.npy')
    p.projection_scatter(
        file_list=file_list,
        plt_labels=[get_dataset_label(f) + '; ' + get_channels_label(f) for f in file_list],
        output_file=f'figures/exp_{projection_algo}_{split}_clf_classes.pdf',
        dataset_file='datasets/clf.csv',
        algorithm=projection_algo,
        split=split,
        color_attribute='class',
        n_neighbors=umap__n_neighbors)

    # print(f'{str(next(cnt_iterator)).zfill(2)} plotting projection colored by r-magnitude')
    # file_list = glob_re(os.path.join(base_dir, 'npy'), f'Xf_{split}_{timestamp}_{backbone}_12_unlabeled(_clf_ft1.npy|.npy)')
    # p.projection_scatter(
    #     file_list=file_list,
    #     plt_labels=['classifier', 'magnitudes regression model'],
    #     output_file=f'figures/exp_{projection_algo}_{split}_clf_magnitudes.pdf',
    #     dataset_file='datasets/clf.csv',
    #     algorithm=projection_algo,
    #     split=split,
    #     color_attribute='r',
    #     n_neighbors=umap__n_neighbors)

    # print(f'{str(next(cnt_iterator)).zfill(2)} plotting X + Xu projection (extracted from classifier) colored by class')
    # file_list = [os.path.join(base_dir, 'npy', f'Xf-Xuf_{split}_{timestamp}_{backbone}_12_unlabeled_clf_ft1.npy')]
    # p.projection_scatter(
    #     file_list=file_list,
    #     plt_labels=['X + Xu features'],
    #     output_file=f'figures/exp_{projection_algo}_{split}_clf_X-Xu.pdf',
    #     dataset_file='datasets/clf.csv',
    #     algorithm=projection_algo,
    #     split=split,
    #     color_attribute='class')

    # print(f'{str(next(cnt_iterator)).zfill(2)} plotting X + Xu projection (extracted from regression model) colored by class')
    # file_list = [os.path.join(base_dir, 'npy', f'Xf-Xuf_{split}_{timestamp}_{backbone}_12_unlabeled.npy')]
    # p.projection_scatter(
    #     file_list=file_list,
    #     plt_labels=['X + Xu features'],
    #     output_file=f'figures/exp_{projection_algo}_{split}_clf_X-Xu_regression-model.pdf',
    #     dataset_file='datasets/clf.csv',
    #     algorithm=projection_algo,
    #     split=split,
    #     color_attribute='class')