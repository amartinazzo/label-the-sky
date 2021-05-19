import _base as b
import gen_plots as p
import numpy as np
import os
import sys
from time import time
import yaml

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

def make_acc_attribute_plots(base_dir, timestamp, backbone, pretraining_dataset, split, attributes):
    p.set_plt_style()
    for attr in attributes:
        attr_name = attr.replace('_', '-')
        p.acc_attribute_bins(
            yhat_files_glob=os.path.join(base_dir, f'npy/yhat_{split}_{timestamp}_{backbone}_*_{pretraining_dataset}_clf_ft1.npy'),
            dataset_file='datasets/clf.csv',
            split=split,
            attribute=attr,
            output_file=f'figures/{timestamp}_acc-{attr_name}_{split}.pdf')
        p.clear_plot()

if __name__ == '__main__':
    b.set_random_seeds()

    if len(sys.argv) < 2:
        print('usage: python {} <yaml_file> <skip_predictions>'.format(sys.argv[0]))
        exit(1)

    config_file = sys.argv[1]
    skip_predictions = bool(int(sys.argv[2])) if sys.argv[2] else False
    config = yaml.load(open(config_file))

    base_dir = os.environ['HOME']
    timestamp = config['timestamp']
    backbone_lst = config['backbones']
    n_channels_lst =config['n_channels']
    pretraining_dataset_lst = config['pretraining_datasets']
    finetune_lst = config['finetune']
    split = config['eval_split']

    # compute feature vectors and predictions
    if not skip_predictions:
        for backbone in backbone_lst:
            for n_channels in n_channels_lst:
                for pretraining_dataset in pretraining_dataset_lst:
                    for finetune in finetune_lst:
                        if (not pretraining_dataset and finetune) or (pretraining_dataset=='imagenet' and n_channels!=3):
                            continue
                        evaluate(base_dir, timestamp, backbone, n_channels, pretraining_dataset, finetune, split)


    # compute plots
    attributes = ['r', 'r_err', 'fwhm']
    for backbone in backbone_lst:
        make_acc_attribute_plots(base_dir, timestamp, backbone, 'unlabeled', split, attributes)
