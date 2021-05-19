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
weights_file = os.path.join(base_dir, 'trained_models', model_name+'_2.h5')

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

print('loading data')
X_test, y_test = trainer.load_data(dataset='clf', split='test')

start = time()

print('predicting on test set')
y_hat, X_features = trainer.extract_features_and_predict(X_test)
print(y_hat.shape, X_features.shape)
np.save(os.path.join('npy', 'yhat_test_'+model_name), y_hat)
np.save(os.path.join('npy', 'Xf_test_'+model_name), X_features)
trainer.evaluate(X_test, y_test)
print('--- minutes taken:', int((time() - start) / 60))

print('generating figures')
# TODO