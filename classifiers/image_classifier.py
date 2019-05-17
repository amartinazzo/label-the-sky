import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import datagen

from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np
import os
import pandas as pd
from resnext import ResNeXt
import sklearn.metrics as metrics


mode = 'train' # train or eval

models_dir = "image-models/"
weights_file = "image-models/resnext-all-mags-12-bands.h5"
save_file = "image-models/resnext-all-mags-12-bands-more-epochs.h5"
home_path = os.path.expanduser('~')
params = {'data_folder': home_path+'/raw-data/crops/normalized/', 'dim': (32,32,12), 'n_classes': 3}

n_classes = 3
n_epoch = 100
img_dim = (32,32,12)
depth = 29
cardinality = 8
width = 16

# make only 1 gpu visible
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='1'

# create resnext model
model = ResNeXt(img_dim, depth=depth, cardinality=cardinality, width=width, classes=n_classes)
print("model created")

# model.summary()

optimizer = Adam(lr=1e-3)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
print("finished compiling")

# load dataset iterators
X_train, _, labels_train = datagen.get_sets(home_path+'/raw-data/matched_cat_dr1_full_train.csv')
X_val, y_true, labels_val = datagen.get_sets(home_path+'/raw-data/matched_cat_dr1_full_val.csv')
print('train size', len(X_train))
print('val size', len(X_val))

train_generator = datagen.DataGenerator(X_train, labels=labels_train, **params)
val_generator = datagen.DataGenerator(X_val, labels=labels_val, **params)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if os.path.exists(weights_file):
    model.load_weights(weights_file)
    print("model loaded.")

# train
if mode=='train':
    lr_reducer = ReduceLROnPlateau(
        monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=10, min_lr=1e-6)
    model_checkpoint = ModelCheckpoint(
      weights_file, monitor="val_acc", save_best_only=True, save_weights_only=True, mode='auto')
    early_stop = EarlyStopping(monitor='val_acc', patience=10)

    callbacks = [lr_reducer, model_checkpoint, early_stop]

    model.fit_generator(
        generator=train_generator,
        validation_data=val_generator,
        epochs=n_epoch,
        callbacks=callbacks,
        verbose=2)

# make inferences on model
print('predicting')
pred_generator = datagen.DataGenerator(X_val, batch_size=len(y_true)//8, shuffle=False, **params)
y_pred = model.predict_generator(pred_generator)
y_pred = np.argmax(y_pred, axis=1)
y_true = y_true[:len(y_pred)]

print('y_true shape', y_true.shape)
print('y_pred shape', y_pred.shape)

# compute accuracy
accuracy = metrics.accuracy_score(y_true, y_pred) * 100
error = 100 - accuracy
print("accuracy : ", accuracy)
print("error : ", error)

# compute confusion matrix
cm = metrics.confusion_matrix(y_true, y_pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('confusion matrix')
print(cm)