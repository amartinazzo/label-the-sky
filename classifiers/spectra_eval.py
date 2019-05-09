from spectra_classifier import build_model
from datagen import DataGenerator
import keras
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit


model_name = 'spectra-models/model_1conv-05-0.27.h5'


class_map = {'GALAXY': 0, 'STAR': 1, 'QSO': 2}
params = {'data_folder': '../../raw-data/spectra/', 'dim': (5500,1), 'n_classes': 3, 'extension': 'txt'}

df = pd.read_csv('../csv/train_val_set_earlydr_spectra.csv')
X = df['id'].values
y = df['class'].apply(lambda c: class_map[c]).values
labels = dict(zip(X, y))
del df

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
train_idx, val_idx = next(sss.split(X,y))

X_val, y_true = X[val_idx], y[val_idx]

val_generator = DataGenerator(X_val, batch_size=len(X_val), shuffle=False, **params)

model = build_model(n_filters=16, kernel_size=100, strides=90, input_shape=params['dim'], n_classes=params['n_classes'])
model.load_weights(model_name)

print('predicting')
y_pred = model.predict_generator(
    val_generator,
    use_multiprocessing=True,
    workers=6,
    verbose=2)

print('saving y_pred')
np.save('spectra-models/preds-model_1conv-05-0.27.npy', y_pred)

# y_pred = np.load('ypred_{}.npy'.format(model_name[:-3]))
y_pred = np.argmax(y_pred, axis=1)
print(y_true[:10])
print(y_pred[:10])
print(y_true.shape, y_pred.shape)

# plot_confusion_matrix(y_true, y_pred, classes=['GALAXY', 'STAR', 'QSO'], normalize=True)
# plt.savefig('confusion_1conv-05-0.27.png')

cm = confusion_matrix(y_true, y_pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

print('confusion matrix')
print(cm)