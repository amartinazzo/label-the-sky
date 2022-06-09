import os
import sys
from time import time

from label_the_sky.training.trainer import Trainer, set_random_seeds

set_random_seeds()

if len(sys.argv) != 8:
    print('usage: python {} <dataset> <backbone> <pretraining_dataset> <n_channels> <finetune> <dataset_mode> <timestamp>'.format(
        sys.argv[0]))
    exit(1)

dataset = sys.argv[1]
backbone = sys.argv[2]
pretraining_dataset = None if sys.argv[3]=='None' else sys.argv[3]
n_channels = int(sys.argv[4])
finetune = True if sys.argv[5]=='1' else False
dataset_mode = sys.argv[6]
timestamp = sys.argv[7]

if dataset_mode not in ['lowdata', 'full']:
    raise Exception('dataset_mode must be: lowdata, full')

base_dir = os.environ['HOME']

if pretraining_dataset is not None and pretraining_dataset!='imagenet':
    weights_file = os.path.join(
        base_dir, 'trained_models', f'{timestamp}_{backbone}_{n_channels}_{pretraining_dataset}.h5')
else:
    weights_file = pretraining_dataset

model_name = f'{timestamp}_{backbone}_{n_channels}_{pretraining_dataset}_clf_ft{int(finetune)}_{dataset_mode}'

trainer = Trainer(
    backbone=backbone,
    n_channels=n_channels,
    output_type='class',
    base_dir=base_dir,
    weights=weights_file,
    model_name=model_name
)

print('loading data')
X_train, y_train = trainer.load_data(dataset=dataset, split='train')
X_val, y_val = trainer.load_data(dataset=dataset, split='val')
X_test, y_test = trainer.load_data(dataset=dataset, split='test')

start = time()

mode = 'top_clf'
if pretraining_dataset is None:
    mode = 'from_scratch'
elif finetune:
    mode = 'finetune'

trainer.describe(verbose=True)

print(f'training: {mode}; dataset mode: {dataset_mode}')
if dataset_mode == 'full':
    trainer.train(X_train, y_train, X_val, y_val, mode=mode)
else:
    trainer.train_lowdata(X_train, y_train, X_val, y_val, mode=mode)

trainer.dump_history('history')
print('--- minutes taken:', int((time() - start) / 60))

if dataset_mode == 'full':
    print('loading best model')
    trainer.load_weights(os.path.join(base_dir, 'trained_models', model_name+'.h5'))

    print('evaluating model on validation set')
    trainer.evaluate(X_val, y_val)
    print('--- minutes taken:', int((time() - start) / 60))
