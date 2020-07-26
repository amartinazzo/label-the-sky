import _base as b
import os
import sys
from time import time

b.set_random_seeds()

if len(sys.argv) != 7:
    print('usage: python {} <dataset> <backbone> <n_channels> <weights> <finetune> <timestamp>'.format(
        sys.argv[0]))
    exit(1)

dataset = sys.argv[1]
n_channels = int(sys.argv[2])
backbone = sys.argv[3]
weights = sys.argv[4]
finetune = True if sys.argv[5]=='1' else False
timestamp = sys.argv[6]

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
X_train, y_train = trainer.load_data(dataset=dataset, split='train')
X_val, y_val = trainer.load_data(dataset=dataset, split='val')
X_test, y_test = trainer.load_data(dataset=dataset, split='test')

start = time()

if weights is None:
    print('training: from scratch')
    trainer.train(X_train, y_train, X_val, y_val)
elif finetune:
    print('training: finetuning')
    trainer.finetune(X_train, y_train, X_val, y_val)
else:
    print('training: top clf')
    trainer.train_top(X_train, y_train, X_val, y_val)

trainer.dump_history()
print('--- minutes taken:', int((time() - start) / 60))

print('evaluating model on test set')
trainer.evaluate(X_test, y_test)
print('--- minutes taken:', int((time() - start) / 60))
