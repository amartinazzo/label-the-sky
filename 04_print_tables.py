import sys
import pandas as pd
from label_the_sky.postprocessing.table import agg_histories

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
    print(dfg.to_latex())
