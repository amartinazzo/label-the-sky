import _base as b
import os
import sys
from time import time

b.set_random_seeds()

if len(sys.argv) != 6:
    print('usage: python {} <backbone> <nbands> <weights> <finetune> <timestamp>'.format(
        sys.argv[0]))
    exit(1)

backbone = sys.argv[1]
n_channels = int(sys.argv[2])
weights = sys.argv[3]
finetune = bool(sys.argv[4])
timestamp = sys.argv[5]

base_dir = os.environ['HOME']

if weights not in b.OUTPUT_TYPES+['imagenet']:
    print('setting weights to NULL')
    weights = None

if weights is not None and weights!='imagenet':
    weights_file = os.path.join(
        base_dir, 'trained_models', f'{timestamp}_{backbone}_{n_channels}_{weights}.h5')
else:
    weights_file = weights

trainer = b.Trainer(
    backbone=backbone,
    n_channels=n_channels,
    output_type='class',
    base_dir=base_dir,
    weights=weights_file,
    model_name=f'{timestamp}_{backbone}_{n_channels}_{weights}_clf_ft{int(finetune)}'
)

print('loading data')
subset = 'clf'
X_train, y_train = trainer.load_data(subset=subset, split='train')
X_val, y_val = trainer.load_data(subset=subset, split='val')
X_test, y_test = trainer.load_data(subset=subset, split='test')

start = time()

print('training model')
if weights is None:
    trainer.train(X_train, y_train, X_val, y_val)
elif finetune:
    trainer.finetune(X_train, y_train, X_val, y_val)
else:
    trainer.train_top(X_train, y_train, X_val, y_val)
trainer.dump_history()
print('--- minutes taken:', int((time() - start) / 60))

print('evaluating model on test set')
trainer.evaluate(X_test, y_test)
print('--- minutes taken:', int((time() - start) / 60))

print('printing history')
trainer.print_history()
