from label_the_sky.config import config
from label_the_sky.postprocessing.table import agg_histories, print_latex
from label_the_sky.training.trainer import MAG_MAX
from label_the_sky.utils import glob_re

backbone = config['eval_backbone']
timestamp = config['timestamp']

cnt_iterator = iter(range(100))

print(f'\n{str(next(cnt_iterator)).zfill(2)} pretraining errors')
history_lst = agg_histories(
    file_list=glob_re('../history', f'{timestamp}_{backbone}_(12|05|03)_unlabeled.json'),
    metric='val_loss',
    mode='min',
    rescale_factor=MAG_MAX)
print_latex(history_lst, cols=['n_channels', 'val_loss'])

print(f'\n{str(next(cnt_iterator)).zfill(2)} channels clf')
history_lst = agg_histories(
    file_list=glob_re('../history', f'{timestamp}_{backbone}_(12|05|03)_unlabeled_clf_ft(0|1).json'),
    metric='val_accuracy',
    mode='max')
print_latex(
    history_lst,
    cols=['n_channels', 'finetune', 'val_accuracy'],
    groupby_cols=['n_channels', 'finetune'])

# df.drop(columns=['timestamp', 'backbone', 'runs'], inplace=True)
# dfg = df.groupby(['weights', 'finetune', 'n_channels']).first().unstack().unstack().T
# print(dfg)
# print(dfg.to_latex())
