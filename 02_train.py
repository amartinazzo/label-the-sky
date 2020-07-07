import _base as b
from pandas import read_csv
import os
import sys
from time import time

b.set_random_seeds()

if len(sys.argv) != 5:
    print('usage: python {} <backbone> <nbands> <weights> <finetune>'.format(
        sys.argv[0]))
    exit(1)

backbone = sys.argv[1]
n_channels = int(sys.argv[2])
weights = sys.argv[3]
finetune = bool(sys.argv[4])

base_dir = os.environ['HOME']
csv_file = os.path.join(base_dir, 'mnt/label-the-sky/csv/dr1_split.csv')

if weights is not None & weights!='imagenet':
    weights = weights.split('_')
    ts = weights[0]
    target = weights[1]
    weights = os.path.join(base_dir, 'trained_models', f'{ts}_{backbone}_{target}_{n_channels}.h5')

print('csv_file', csv_file)
print('backbone', backbone)
print('n_channels', n_channels)

trainer = b.Trainer(
    backbone=backbone,
    n_channels=n_channels,
    output_type='class',
    base_dir=base_dir,
    weights=weights
)

print('building datasets')
df = read_csv(csv_file)
df = df[~df.pretraining]
gen_train, y_train = trainer.build_dataset(df, split='train')
gen_val, y_val = trainer.build_dataset(df, split='val')
gen_test, y_test = trainer.build_dataset(df, split='test')

start = time()

print('training model')
trainer.build_model()
if finetune:
    trainer.finetune(gen_train, gen_val, epochs=1)
else:
    trainer.train_clf(gen_train, y_train, gen_val, y_val, epochs=1)
trainer.dump_history()
print('--- minutes taken:', int((time() - start) / 60))

print('evaluating model on test set')
trainer.evaluate(gen_test, y_test)
print('--- minutes taken:', int((time() - start) / 60))

print('printing history')
trainer.print_history()
