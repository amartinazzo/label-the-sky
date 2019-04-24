from datagen import DataGenerator
import numpy as np
import pandas as pd
from keras.models import Sequential
from sklearn.model_selection import StratifiedShuffleSplit
import os


os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='1'

class_map = {'GALAXY': 0, 'STAR': 1, 'QSO': 2}
params = {'data_folder': '../../raw-data/spectra-processed/', 'dim': (5500,), 'batch_size': 256, 'n_classes': 3}

# load dataset iterators
df = pd.read_csv('train_val_set.csv')
X = df['id'].values
y = df['class'].apply(lambda c: class_map[c]).values
labels = dict(zip(X, y))
del df

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
train_idx, val_idx = sss.split(X,y)

X_train, X_val = X[train_idx], X[val_idx]
print('train size', len(X_train))
print('val size', len(X_val))

train_generator = DataGenerator(X_train, labels, **params)
val_generator = DataGenerator(X_val, labels, **params)

# create model
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=20, strides=15, activation='relu', input_shape=params['dim']))
model.add(Conv1D(filters=64, kernel_size=20, strides=15, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(filters=128, kernel_size=20, strides=15, activation='relu'))
model.add(Conv1D(filters=128, kernel_size=20, strides=15, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(params['n_classes'], activation='softmax'))
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='acc', patience=1)
]

# train
model.fit_generator(
	generator=train_generator,
    validation_data=val_generator,
    use_multiprocessing=True,
    workers=6)