import _base as b
import os
import sys
from time import time

b.set_random_seeds()

if len(sys.argv) != 2:
    print('usage: python {} <timestamp>'.format(
        sys.argv[0]))
    exit(1)

timestamp = sys.argv[1]

trainer = b.Trainer(
    backbone=None,
    n_channels=12,
    output_type='class',
    base_dir=base_dir,
    weights=None,
    model_name=f'{timestamp}_catalog'
)

print('loading magnitudes data')
X_train, y_train = trainer.load_magnitudes(split='train')
X_val, y_val = trainer.load_magnitudes(split='val')
X_test, y_test = trainer.load_magnitudes(split='test')

start = time()

trainer.train_catalog(X_train, y_train, X_val, y_val)
trainer.dump_history()
print('--- minutes taken:', int((time() - start) / 60))

print('evaluating model on test set')
trainer.evaluate(X_test, y_test)
print('--- minutes taken:', int((time() - start) / 60))

print('printing history')
trainer.print_history()
