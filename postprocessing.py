from glob import glob
import json
import pandas as pd
import numpy as np
import sys


AGG_FN = {
    'max': lambda arr: np.max(arr).round(4),
    'mean': lambda arr: str(
        np.mean(arr).round(4)) + '+-' + str(
        np.std(arr, ddof=1).round(4)) if np.mean(arr) < 1e3 else 'inf'
}


def agg_histories(pattern, mode):
    agg_fn = AGG_FN.get(mode)
    hist_list = []
    files = glob(pattern)
    for f in files:
        split_str = f.split('/')[-1][:-5].split('_')
        timestamp = split_str[0]
        backbone = split_str[1]
        n_channels = split_str[2]
        weights = split_str[3]
        finetune = split_str[5][-1]
        with open(f) as json_file:
            data = json.load(json_file)
            if type(data) == list:
                val_acc = []
                val_loss = []
                for hist in data:
                    arg = np.argmax(hist['val_accuracy'])
                    val_acc.append(hist['val_accuracy'][arg])
                    val_loss.append(hist['val_loss'][arg])
                hist_data = {
                    'val_acc': agg_fn(val_acc),
                    'val_loss': agg_fn(val_loss),
                    'runs': len(val_acc)
                }
            else:
                arg = np.argmax(data['val_accuracy'])
                hist_data = {
                    'val_acc': data['val_accuracy'][arg],
                    'val_loss': data['val_loss'][arg],
                    'runs': 1
                }
            hist_data.update({
                'timestamp': timestamp,
                'backbone': backbone,
                'n_channels': n_channels,
                'weights': weights,
                'finetune': finetune
            })
            hist_list.append(hist_data)
    return hist_list


def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: python {} "<glob_pattern>" <mode>'.format(sys.argv[0]))
        exit()

    glob_pattern = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv)>2 else 'mean'

    history_dict = agg_histories(glob_pattern, mode=mode)

    df = pd.DataFrame.from_dict(history_dict)
    df.drop(columns=['timestamp', 'backbone', 'runs'], inplace=True)
    dfg = df.groupby(['weights', 'finetune', 'n_channels']).first().unstack().unstack().T
    print(dfg)
