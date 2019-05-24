import os,sys,inspect
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))
import datagen
from keras.callbacks import ModelCheckpoint, EarlyStopping
from models import _1d_conv_net, dense_net
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


mode = 'eval' # train or eval
filters = None #{'r': [16, 19]}
width = 64

weights_file = 'classifiers/spectra-models/dense_all-mags_64.h5'
filter_str = 'all-mags' if filters is None else 'mag{}-{}'.format(filters['r'][0], filters['r'][1])
save_file = 'classifiers/spectra-models/dense_{}_{}.h5'.format(filter_str, width)

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'

home_path = os.path.expanduser('~')

class_map = {'GALAXY': 0, 'STAR': 1, 'QSO': 2}

params = {
    'data_folder': home_path+'/raw-data/spectra/',
    'dim': (5500,),
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

# model = _1d_conv_net(
#     n_filters=16, kernel_size=100, strides=90, input_shape=params['dim'], n_classes=params['n_classes'])

model = dense_net(input_shape=params['dim'], width=width)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
if weights_file is not None:
    model.load_weights(weights_file)

# train

if mode=='train':
    callbacks_list = [
        ModelCheckpoint(filepath=save_file, monitor='val_loss', save_best_only=True),
        EarlyStopping(monitor='val_acc', patience=5)
    ]

    model.fit_generator(
    	epochs=30,
        callbacks=callbacks_list,
    	generator=train_generator,
        validation_data=val_generator,
        verbose=2)

# evaluate

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