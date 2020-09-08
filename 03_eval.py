import _base as b
import numpy as np
import os
import sys
from time import time

b.set_random_seeds()

if len(sys.argv) != 6:
    print('usage: python {} <timestamp> <backbone> <n_channels> <pretraining_dataset> <finetune>'.format(
        sys.argv[0]))
    exit(1)

timestamp = sys.argv[1]
backbone = sys.argv[2]
n_channels = int(sys.argv[3])
pretraining_dataset = sys.argv[4]
finetune = True if sys.argv[5]=='1' else False

base_dir = os.environ['HOME']
model_name = f'{timestamp}_{backbone}_{n_channels}_{pretraining_dataset}_clf_ft{int(finetune)}'
weights_file = os.path.join(base_dir, 'trained_models', model_name+'.h5')

trainer = b.Trainer(
    backbone=backbone,
    n_channels=n_channels,
    output_type='class',
    base_dir=base_dir,
    weights=weights_file
)

print('loading data')
X_val, y_val = trainer.load_data(dataset='clf', split='val')

start = time()

print('predicting on val set')
y_hat, X_1024 = trainer.extract_features_and_predict(X_val)
np.save(os.path.join('npy', 'yhat_val_'+model_name))
print('--- minutes taken:', int((time() - start) / 60))

print('generating UMAP embeddings')