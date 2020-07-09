from glob import glob
import json
import pandas as pd
import numpy as np
import sys


def agg_histories(pattern):
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
                    'val_acc': str(
                        np.mean(val_acc).round(4)) + '+-' + str(
                        np.std(val_acc, ddof=1).round(4)),
                    'val_loss': str(
                        np.mean(val_loss).round(4)) + '+-' + str(
                        np.std(val_loss, ddof=1).round(4)) if np.mean(val_loss) < 1e3 else 'inf',
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


if len(sys.argv) != 2:
    print('usage: python {} "<glob_pattern>"'.format(sys.argv[0]))
    exit()

glob_pattern = sys.argv[1]

history_dict = agg_histories(glob_pattern)

df = pd.DataFrame.from_dict(history_dict)
df.drop(columns=['timestamp', 'backbone', 'runs'], inplace=True)
dfg = df.groupby(['weights', 'finetune', 'n_channels']).first().unstack().unstack().T
print(dfg)
