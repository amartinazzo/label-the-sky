from datagen import DataGenerator
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Dropout, Flatten, GlobalAveragePooling1D, MaxPooling1D
from sklearn.model_selection import StratifiedShuffleSplit
import os


def build_model(n_filters, kernel_size, strides, input_shape, n_classes):
    model = Sequential()
    model.add(Conv1D(
        filters=n_filters, kernel_size=kernel_size, strides=strides, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Flatten())
    model.add(Dense(n_classes, activation='softmax'))
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


if __name__=='__main__':
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']='1'

    home_path = os.path.expanduser('~')

    class_map = {'GALAXY': 0, 'STAR': 1, 'QSO': 2}
    params = {'data_folder': home_path+'/raw-data/spectra/', 'dim': (5500,1), 'batch_size': 256, 'n_classes': 3, 'extension': 'txt'}

    # load dataset iterators
    df = pd.read_csv(home_path+'/label-the-sky/csv/train_val_set_earlydr_spectra.csv')
    X = df['id'].values
    y = df['class'].apply(lambda c: class_map[c]).values
    labels = dict(zip(X, y))
    del df

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_idx, val_idx = next(sss.split(X,y))

    X_train, X_val = X[train_idx], X[val_idx]
    print('train size', len(X_train))
    print('val size', len(X_val))

    train_generator = DataGenerator(X_train, labels, **params)
    val_generator = DataGenerator(X_val, labels, **params)

    # create model
    model = build_model(n_filters=16, kernel_size=100, strides=90, input_shape=params['dim'], n_classes=params['n_classes'])

    callbacks_list = [
        ModelCheckpoint(
            filepath='spectra-models/model-{epoch:02d}-{val_loss:.2f}.h5',
            monitor='val_loss', save_best_only=True),
        EarlyStopping(monitor='acc', patience=5)
    ]

    # train
    model.fit_generator(
    	epochs=30,
        callbacks=callbacks_list,
    	generator=train_generator,
        validation_data=val_generator,
        use_multiprocessing=True,
        workers=6,
        verbose=2)
