import os,sys,inspect
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))
import datagen
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from models import _1d_conv_net, dense_net
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix


mode = 'train' # train or eval
filters = None #{'r': [16, 19]}

weights_file = None #'classifiers/spectra-models/model_1conv.h5'
filter_str = 'all-mags' if filters is None else 'mag{}-{}'.format(filters['r'][0], filters['r'][1])
save_file = 'classifiers/spectra-models/conv1d_kernel100_stride90.h5'

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'

home_path = os.path.expanduser('~')

class_map = {'GALAXY': 0, 'STAR': 1, 'QSO': 2}

params = {
    'data_folder': home_path+'/raw-data/spectra/',
    'dim': (5500,1),
    'extension': 'txt',
    'n_classes': 3
}


# load dataset iterators

X_train, _, labels_train = datagen.get_sets('csv/matched_cat_early-dr_filtered_train.csv', filters=filters)
X_val, y_true, labels_val = datagen.get_sets('csv/matched_cat_early-dr_filtered_val.csv', filters=filters)
print('train size', len(X_train))
print('val size', len(X_val))

train_generator = datagen.DataGenerator(X_train, labels=labels_train, **params)
val_generator = datagen.DataGenerator(X_val, labels=labels_val, **params)

# create model

model = _1d_conv_net(
    n_filters=16, kernel_size=100, strides=90, input_shape=params['dim'], n_classes=params['n_classes'])

# model = dense_net(input_shape=params['dim'], width=width)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
if weights_file is not None:
    model.load_weights(weights_file)

# train

if mode=='train':
    callbacks_list = [
        ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=10, min_lr=1e-6),
        ModelCheckpoint(filepath=save_file, monitor='val_loss', save_best_only=True),
        EarlyStopping(monitor='val_acc', patience=10)
    ]

    history = model.fit_generator(
    	epochs=100,
        steps_per_epoch=len(X_train)//128,
        callbacks=callbacks_list,
    	generator=train_generator,
        validation_data=val_generator,
        validation_steps=len(X_val)//128,
        verbose=2)

    print(history)

elif mode=='eval-mags':
    mag_min = 9
    mag_max = 23
    bins = 2 # number of bins between two consecutive integers, e.g. bins=4 means [1.0, 1.25, 1.5, 1.75,]
    intervals = np.linspace(mag_min, mag_max, bins*(mag_max-mag_min)+1)
    mags = []
    acc0 = []
    acc1 = []
    acc2 = []
    for ix in range(len(intervals)-1):
        filters = {'r': (intervals[ix], intervals[ix+1])}
        X_val, y_true, _ = datagen.get_sets('csv/matched_cat_early-dr_filtered_val.csv', filters=filters)
        if X_val.shape[0] == 0:
            continue
        print('predicting for [{}, {}]'.format(intervals[ix], intervals[ix+1]))
        pred_generator = datagen.DataGenerator(X_val, shuffle=False, batch_size=1, **params)
        y_pred = model.predict_generator(pred_generator, steps=len(y_true))
        # print(y_pred[:10])
        y_pred = np.argmax(y_pred, axis=1)

        # compute accuracy
        accuracy = accuracy_score(y_true, y_pred) * 100
        print('accuracy : ', accuracy)

        # compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('confusion matrix')
        print(cm)

        if cm.shape[0]==3:
            mags.append(intervals[ix])
            acc0.append(cm[0,0])
            acc1.append(cm[1,1])
            acc2.append(cm[2,2])

    print('magnitudes', mags)
    print('accuracies for class 0', acc0)
    print('accuracies for class 1', acc1)
    print('accuracies for class 2', acc2)

# evaluate

else:
    print('predicting')
    pred_generator = datagen.DataGenerator(X_val, batch_size=len(y_true)//8, shuffle=False, **params)
    y_pred = model.predict_generator(pred_generator, verbose=2)

    y_pred = np.argmax(y_pred, axis=1)
    y_true = y_true[:len(y_pred)]
    print(y_true[:10])
    print(y_pred[:10])
    print(y_true.shape, y_pred.shape)

    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print('confusion matrix')
    print(cm)