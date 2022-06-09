import os
import sys
from time import time

from label_the_sky.training.trainer import Trainer, set_random_seeds

set_random_seeds()

if len(sys.argv) != 6:
    print('usage: python {} <dataset> <backbone> <n_channels> <target> <timestamp>'.format(
        sys.argv[0]))
    exit(1)

dataset = sys.argv[1]
backbone = sys.argv[2]
n_channels = int(sys.argv[3])
target = sys.argv[4]
timestamp = sys.argv[5]

base_dir = os.environ['HOME']

trainer = Trainer(
    backbone=backbone,
    n_channels=n_channels,
    output_type=target,
    base_dir=base_dir,
    weights=None,
    model_name=f'{timestamp}_{backbone}_{n_channels}_{dataset}'
)

trainer.describe(verbose=True)

print('loading data')
X_train, y_train = trainer.load_data(dataset=dataset, split='train')
X_val, y_val = trainer.load_data(dataset=dataset, split='val')
X_test, y_test = trainer.load_data(dataset=dataset, split='test')

start = time()

print('pretraining model')
trainer.train(X_train, y_train, X_val, y_val, epochs=100)
trainer.dump_history('history')
print('--- minutes taken:', int((time() - start) / 60))

print('evaluating model on validation set')
trainer.pick_best_model()
trainer.evaluate(X_val, y_val)
print('--- minutes taken:', int((time() - start) / 60))

print('printing history')
trainer.print_history()
