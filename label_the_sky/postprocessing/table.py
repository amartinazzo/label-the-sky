import json
import numpy as np
import pandas as pd

from label_the_sky.config import BANDS
from label_the_sky.training.trainer import MAG_MAX
from label_the_sky.utils import get_channels_label


AGG_FN = {
    'max': lambda arr: np.max(arr).round(4),
    'mean': lambda arr: str(
        np.mean(arr).round(4)) + ' ± ' + str(
        np.std(arr, ddof=1).round(4)) if np.mean(arr) < 1e3 else 'inf',
    'min': lambda arr: np.min(arr).round(4)
}


def agg_histories(file_list, metric='val_loss', mode='min', agg_mode='mean', rescale_factor=1):
    agg_fn = AGG_FN.get(agg_mode)
    run_fn = np.max if mode=='max' else np.min
    history_lst = []
    for f in file_list:
        split_str = f.split('/')[-1][:-5].split('_')
        timestamp = split_str[0]
        backbone = split_str[1]
        n_channels = split_str[2]
        weights = split_str[3]
        finetune = split_str[5][-1] if len(split_str) > 4 else None
        with open(f) as json_file:
            history = json.load(json_file)
            if type(history) == dict:
                history = [history]
            if type(history[0]) == list:
                lst = [run_fn(history_run[-1][metric])*rescale_factor for history_run in history]
            else:
                lst = [run_fn(history_run[metric])*rescale_factor for history_run in history]
            history_data = {
                f'{metric}': agg_fn(lst),
                'runs': len(lst)
            }
            history_data.update({
                'timestamp': timestamp,
                'backbone': backbone,
                'n_channels': n_channels,
                'weights': weights,
                'finetune': finetune
            })
            history_lst.append(history_data)
    return history_lst

def get_df_latex(df, index=False):
    latex_str = df.to_latex(index=index)
    latex_str = latex_str.replace('\\toprule\n', '')
    latex_str = latex_str.replace('\\bottomrule\n', '')
    latex_str = latex_str.replace('\\midrule', '\\hline')
    return latex_str

def print_history_latex(history_lst, cols, groupby_cols=None):
    df = pd.DataFrame.from_dict(history_lst)
    df = df[cols]
    if groupby_cols:
        df = df.groupby(groupby_cols).first()
        for _ in range(len(groupby_cols)-1):
            df = df.unstack()
    else:
        df.sort_values(by=df.columns[0], inplace=True)
    index = True if groupby_cols else False
    print(get_df_latex(df, index=index))

def print_yhat(npy_file, dataset_file, split='test'):
    df = pd.read_csv(dataset_file)
    df = df[df.split==split]
    y_hat = np.load(npy_file)
    y = df[BANDS].values
    y_error = df[[band+'_err' for band in BANDS]].values
    diff = np.abs(y - y_hat)

    df_out = pd.DataFrame()
    df_out['channel'] = BANDS
    df_out['mean absolute error'] = np.mean(diff, axis=0).round(4)
    df_out['mean uncertainty'] = np.mean(y_error, axis=0).round(4)

    print(get_df_latex(df_out))