import os,sys,inspect
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))
import datagen
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np
import os
import pandas as pd
from models import resnext
import sklearn.metrics as metrics


mode = 'eval-mags' # train or eval-mags

n_classes = 3
n_epoch = 100
img_dim = (32,32,12)

# images are stored in float64 (8 bytes per pixel)
# one image is 32*32*12*8/10e6 ~= 0.1 MB
# 80 images are roughly 7.8 MB - fit into two Tesla K20m
batch_size = 80
depth = 29
cardinality = 8
width = 16

models_dir = 'classifiers/image-models/'
weights_file = 'classifiers/image-models/resnext-05-08.h5'
# save_file = 'classifiers/image-models/resnext-all-mags-12-bands.h5'
home_path = os.path.expanduser('~')
weights_file = home_path+'/classifiers/image-models/resnext-05-08.h5'
params = {'data_folder': home_path+'/raw-data/crops/normalized/', 'dim': (32,32,12), 'n_classes': 3, 'batch_size': batch_size}
class_weights = {0: 2, 1: 2.5, 2: 10} # 1/class_proportion

# make only 1 gpu visible
# os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES']='1'

# create resnext model
model = resnext(img_dim, depth=depth, cardinality=cardinality, width=width, classes=n_classes)
print('model created')

model.summary()

optimizer = Adam(lr=1e-3)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
print('finished compiling')

# load dataset iterators
X_train, _, labels_train = datagen.get_sets(home_path+'/raw-data/dr1_full_train.csv')
X_val, y_true, labels_val = datagen.get_sets(home_path+'/raw-data/dr1_full_val.csv')
print('train size', len(X_train))
print('val size', len(X_val))

train_generator = datagen.DataGenerator(X_train, labels=labels_train, **params)
val_generator = datagen.DataGenerator(X_val, labels=labels_val, **params)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if os.path.exists(weights_file):
   model.load_weights(weights_file)
   print('model loaded.')

# train
if mode=='train':
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=10, min_lr=1e-6),
        ModelCheckpoint(save_file, monitor='val_acc', save_best_only=True, save_weights_only=True, mode='auto'),
        #EarlyStopping(monitor='val_acc', patience=10)
    ]

    model.fit_generator(
        steps_per_epoch=len(X_train)//batch_size,
        generator=train_generator,
        validation_data=val_generator,
        validation_steps=len(X_val)//batch_size,
        epochs=n_epoch,
        callbacks=callbacks,
        verbose=2)

elif mode=='eval-mags':
    mag_min = 9
    mag_max = 23
    intervals = np.linspace(mag_min, mag_max, 2*(mag_max-mag_min)+1)
    acc0 = []
    acc1 = []
    acc2 = []
    for ix in range(len(intervals)-1):
        filters = {'r': (intervals[ix], intervals[ix+1])}
        X_val, y_true, _ = datagen.get_sets(home_path+'/raw-data/dr1_full_val.csv', filters=filters)
        print('predicting for [{}, {}]'.format(intervals[ix], intervals[ix+1]))
        pred_generator = datagen.DataGenerator(X_val, shuffle=False, **params)
        print('loaded generator')
        print('generator len', len(pred_generator))
        y_pred = model.predict_generator(pred_generator, steps=np.maximum(len(y_true)//batch_size,1))
        y_pred = np.argmax(y_pred, axis=1)
        print('predicted')
        y_pred = y_pred[:len(y_true)]
        print('y_true shape', y_true.shape)
        print('y_pred shape', y_pred.shape)
        # y_true = y_true[:len(y_pred)]

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

        acc0.append(cm[0,0])
        acc1.append(cm[1,1])
        acc2.append(cm[2,2])

    print('accuracies for class 0', acc0)
    print('accuracies for class 1', acc1)
    print('accuracies for class 2', acc2)

else:
    # make inferences on model
    print('predicting')
    pred_generator = datagen.DataGenerator(X_val, shuffle=False, **params)
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
