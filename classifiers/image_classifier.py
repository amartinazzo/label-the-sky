from datagen import DataGenerator
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np
import os
import pandas as pd
from resnext import ResNeXt
import sklearn.metrics as metrics
from sklearn.model_selection import StratifiedShuffleSplit



mode = 'train' # train or eval

models_dir = "image-models/"
weights_file = "image-models/resnext-5-bands.h5"
class_map = {'GALAXY': 0, 'STAR': 1, 'QSO': 2}
home_path = os.path.expanduser('~')
params = {'data_folder': home_path+'/raw-data/crops/normalized/', 'dim': (32,32,12), 'n_classes': 3, 'bands': [0,5,7,9,11]}

n_classes = 3
n_epoch = 100
img_dim = (32,32,5)
depth = 29
cardinality = 8
width = 16

# make only 1 gpu visible
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='1'

# create resnext model
model = ResNeXt(img_dim, depth=depth, cardinality=cardinality, width=width, classes=n_classes)
print("model created")

model.summary()

optimizer = Adam(lr=1e-3)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
print("finished compiling")

# load dataset iterators
df = pd.read_csv(home_path+'/raw-data/trainval_set_mag16-19.csv')
X = df['id'].values
y = df['class'].apply(lambda c: class_map[c]).values
labels = dict(zip(X, y))
del df

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
train_idx, val_idx = next(sss.split(X,y))

X_train, X_val, y_true = X[train_idx], X[val_idx], y[val_idx]
print('train size', len(X_train))
print('val size', len(X_val))

train_generator = DataGenerator(X_train, labels=labels, **params)
val_generator = DataGenerator(X_val, labels=labels, **params)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if os.path.exists(weights_file):
    model.load_weights(weights_file)
    print("model loaded.")

if mode=='train':
    # set model checkpoints
    lr_reducer = ReduceLROnPlateau(
        monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=10, min_lr=1e-6)

    model_checkpoint = ModelCheckpoint(
      weights_file, monitor="val_acc", save_best_only=True, save_weights_only=True, mode='auto')

    callbacks = [lr_reducer, model_checkpoint]

    # train model
    model.fit_generator(
        generator=train_generator,
        validation_data=val_generator,
        epochs=n_epoch,
        callbacks=callbacks,
        verbose=2)

# make inferences on model
pred_generator = DataGenerator(X_val, batch_size=len(y_true)//8, shuffle=False, **params)
y_pred = model.predict_generator(pred_generator)
y_pred = np.argmax(y_pred, axis=1)

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