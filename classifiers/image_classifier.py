import os,sys,inspect
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))
from datagen import DataGenerator
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import numpy as np
import os
import pandas as pd
from models import resnext
import sklearn.metrics as metrics
import utils


mode = 'train' # train or eval-mags

task = 'regression' # classification or regression (magnitudes)
n_classes = 12
n_epoch = 300
img_dim = (32,32,12)

# images are stored in float64 (8 bytes per pixel)
# one image is 32*32*12*8/10e6 ~= 0.1 MB
batch_size = 256
cardinality = 4
data_mode = 'magnitudes' #classifier or magnitudes
depth = 29
loss = 'mean_absolute_error' #'categorical_crossentropy'
width = 16


models_dir = 'classifiers/image-models/'
weights_file = None #'classifiers/image-models/resnext_depth11_card4_300epc.h5'
save_file = 'classifiers/image-models/resnext_depth29_card4_300eph_asinh_regression.h5'
#'classifiers/image-models/resnext_depth29_card4_300epc_regression_smallset.h5'

home_path = os.path.expanduser('~')
params = {'data_folder': home_path+'/raw-data/crops_asinh_1fwhm/',
            'dim': (32,32,12), 'n_classes': n_classes, 'mode': data_mode}
class_weights = {0: 1, 1: 1.25, 2: 5} # 1/class_proportion

# make only 1 gpu visible
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'

###

# create resnext model
model = resnext(img_dim, depth=depth, cardinality=cardinality, width=width, classes=n_classes)
print('model created')

# model.summary()

optimizer = Adam(lr=1e-4)
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
print('finished compiling')

# load dataset iterators
X_train, _, labels_train = utils.get_sets('csv/dr1_flag0_ndet12_train.csv', mode=data_mode)
X_val, y_true, labels_val = utils.get_sets('csv/dr1_flag0_ndet12_val.csv', mode=data_mode)
print('train size', len(X_train))
print('val size', len(X_val))

train_generator = DataGenerator(X_train, labels=labels_train, batch_size=batch_size, **params)
val_generator = DataGenerator(X_val, labels=labels_val, batch_size=batch_size, **params)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if weights_file is not None and os.path.exists(weights_file):
   model.load_weights(weights_file)
   print('model weights loaded!')

# train
if mode=='train':
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=10, min_lr=1e-6),
        ModelCheckpoint(save_file, monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min'),
        EarlyStopping(monitor='val_loss', patience=20)
    ]

    history = model.fit_generator(
        steps_per_epoch=len(X_train)//batch_size,
        generator=train_generator,
        validation_data=val_generator,
        validation_steps=len(X_val)//batch_size,
        epochs=n_epoch,
        callbacks=callbacks,
        verbose=2)

    print('loss', history.history['loss'])
    print('val_loss', history.history['val_loss'])

    if task=='classification':
        print('predicting')
        model.load_weights(save_file)
        pred_generator = DataGenerator(X_val, shuffle=False, **params)
        y_pred = model.predict_generator(pred_generator, steps=len(y_true)//batch_size)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = y_true[:len(y_pred)]

        print('y_true shape', y_true.shape)
        print('y_pred shape', y_pred.shape)

        # compute accuracy
        accuracy = metrics.accuracy_score(y_true, y_pred) * 100
        error = 100 - accuracy
        print('accuracy : ', accuracy)
        print('error : ', error)

        # compute confusion matrix
        cm = metrics.confusion_matrix(y_true, y_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('confusion matrix')
        print(cm)

elif mode=='eval-mags':
    mag_min = 9
    mag_max = 23
    bins = 4 # number of bins between two consecutive integers, e.g. bins=4 means [1.0, 1.25, 1.5, 1.75,]
    intervals = np.linspace(mag_min, mag_max, bins*(mag_max-mag_min)+1)
    mags = []
    acc0 = []
    acc1 = []
    acc2 = []
    for ix in range(len(intervals)-1):
        filters = {'r': (intervals[ix], intervals[ix+1])}
        X_val, y_true, _ = utils.get_sets(home_path+'/raw-data/dr1_full_val.csv', filters=filters)
        if X_val.shape[0] == 0:
            continue
        print('predicting for [{}, {}]'.format(intervals[ix], intervals[ix+1]))
        pred_generator = DataGenerator(X_val, shuffle=False, batch_size=1, **params)
        y_pred = model.predict_generator(pred_generator)
        # print(y_pred[:10])
        y_pred = np.argmax(y_pred, axis=1)

        # compute accuracy
        accuracy = metrics.accuracy_score(y_true, y_pred) * 100
        error = 100 - accuracy
        print('accuracy : ', accuracy)
        print('error : ', error)

        # compute confusion matrix
        cm = metrics.confusion_matrix(y_true, y_pred)
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

# else:
#     # make inferences on model
#     print('predicting')
#     X_val, y_true, _ = utils.get_sets(home_path+'/raw-data/dr1_full_val.csv', filters={'r': (21,22)})
#     idx = list(X_val).index('SPLUS.STRIPE82-0033.05627')
#     X_val = [X_val[idx]]
#     y_true = [y_true[idx]]
#     print(X_val, y_true)
#     pred_generator = DataGenerator(X_val, shuffle=False, batch_size=1, **params)
#     y_pred = model.predict_generator(pred_generator, steps=len(y_true))
#     print(y_pred)
#     y_pred = np.argmax(y_pred, axis=1)

#     preds_correct = y_pred==y_true
#     x_miss = X_val[~preds_correct]
#     print('missclasified', len(x_miss))
#     print(x_miss)

#     # compute accuracy
#     accuracy = metrics.accuracy_score(y_true, y_pred) * 100
#     print('accuracy : ', accuracy)

#     # compute confusion matrix
#     cm = metrics.confusion_matrix(y_true, y_pred)
#     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#     print('confusion matrix')
#     print(cm)
