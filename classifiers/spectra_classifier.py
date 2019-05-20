import os,sys,inspect
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))
import datagen
from models import dense_net
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Dropout, Flatten, GlobalAveragePooling1D, MaxPooling1D
from models import _1d_conv_net
from sklearn.metrics import confusion_matrix
import os


mode = 'train' # train or eval
weights_file = None #'spectra-models/model_1conv-05-0.27.h5'
save_file = 'classifiers/spectra-models/dense-64_05-20.h5'

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='1'

home_path = os.path.expanduser('~')

class_map = {'GALAXY': 0, 'STAR': 1, 'QSO': 2}

params = {
    'batch_size': 256,
    'data_folder': home_path+'/raw-data/spectra/',
    'dim': (5500,),
    'extension': 'txt',
    'n_classes': 3
}


# load dataset iterators

X_train, _, labels_train = datagen.get_sets('csv/matched_cat_early-dr_filtered_train.csv')
X_val, y_true, labels_val = datagen.get_sets('csv/matched_cat_early-dr_filtered_val.csv')
print('train size', len(X_train))
print('val size', len(X_val))

train_generator = datagen.DataGenerator(X_train, labels=labels_train, **params)
val_generator = datagen.DataGenerator(X_val, labels=labels_val, **params)

# create model

# model = _1d_conv_net(
#     n_filters=16, kernel_size=100, strides=90, input_shape=params['dim'], n_classes=params['n_classes'])

model = dense_net(input_shape=params['dim'], width=64)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
if weights_file is not None:
    model.load_weights(model_name)

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
        use_multiprocessing=True,
        workers=6,
        verbose=2)

# evaluate

print('predicting')
y_pred = model.predict_generator(
    val_generator,
    use_multiprocessing=True,
    workers=6,
    verbose=2)

y_pred = np.argmax(y_pred, axis=1)
print(y_true[:10])
print(y_pred[:10])
print(y_true.shape, y_pred.shape)

cm = confusion_matrix(y_true, y_pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

print('confusion matrix')
print(cm)