import json
import numpy as np
import pandas as pd

from label_the_sky.training.trainer import MAG_MAX


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

def print_latex(history_lst, cols, groupby_cols=None):
    df = pd.DataFrame.from_dict(history_lst)
    df = df[cols]
    if groupby_cols:
        df = df.groupby(groupby_cols).first()
        for _ in range(len(groupby_cols)-1):
            df = df.unstack()
    index = True if groupby_cols else False
    print(df.to_latex(index=index))
