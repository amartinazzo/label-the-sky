import os

from label_the_sky.config import config
from label_the_sky.postprocessing.table import agg_histories, print_history_latex, print_yhat
from label_the_sky.training.trainer import MAG_MAX
from label_the_sky.utils import glob_re

backbone = config['eval_backbone']
timestamp = config['timestamp']

cnt_iterator = iter(range(100))

print(f'\n{str(next(cnt_iterator)).zfill(2)} pretraining validation errors\n')
history_lst = agg_histories(
    file_list=glob_re('../history', f'{timestamp}_{backbone}_(12|05|03)_unlabeled.json'),
    metric='val_loss',
    mode='min',
    rescale_factor=MAG_MAX)
print_history_latex(history_lst, cols=['n_channels', 'val_loss'])

print(f'\n{str(next(cnt_iterator)).zfill(2)} pretraining test errors per channel (12 channel model)\n')
print_yhat(
    npy_file=os.path.join(os.environ['HOME'], 'npy', 'yhat_unlabeled-test_0206_vgg_12_unlabeled.npy'),
    dataset_file='datasets/unlabeled.csv',
    split='test')

print(f'\n{str(next(cnt_iterator)).zfill(2)} RGB clf\n')
history_lst = agg_histories(
    file_list=glob_re('../history', f'{timestamp}_{backbone}_03_(imagenet|unlabeled)_clf_ft(0|1).json'),
    metric='val_accuracy',
    mode='max')
print_history_latex(
    history_lst,
    cols=['weights', 'finetune', 'val_accuracy'],
    groupby_cols=['weights', 'finetune'])

print(f'\n{str(next(cnt_iterator)).zfill(2)} channels clf\n')
history_lst = agg_histories(
    file_list=glob_re('../history', f'{timestamp}_{backbone}_(12|05|03)_unlabeled_clf_ft(0|1).json'),
    metric='val_accuracy',
    mode='max')
print_history_latex(
    history_lst,
    cols=['n_channels', 'finetune', 'val_accuracy'],
    groupby_cols=['n_channels', 'finetune'])

print(f'\n{str(next(cnt_iterator)).zfill(2)} channels clf lowdata\n')
history_lst = agg_histories(
    file_list=glob_re('../history', f'{timestamp}_{backbone}_(12|05|03)_unlabeled_clf_ft(0|1)_lowdata.json'),
    metric='val_accuracy',
    mode='max')
print_history_latex(
    history_lst,
    cols=['n_channels', 'finetune', 'val_accuracy'],
    groupby_cols=['n_channels', 'finetune'])
