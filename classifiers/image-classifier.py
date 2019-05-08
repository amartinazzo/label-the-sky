import numpy as np
import sklearn.metrics as metrics
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils import np_utils
from resnext import ResNeXt
import os

batch_size = 100
n_classes = 3
n_epoch = 100

img_dim = (32,32,12)
depth = 29
cardinality = 8
width = 16

models_dir = "image-models/"
weights_file = "image-models/resnext-05-08.h5"
class_map = {'GALAXY': 0, 'STAR': 1, 'QSO': 2}
params = {'data_folder': '../../raw-data/crops/normalized', 'dim': img_dim, 'batch_size': 256, 'n_classes': 3}

# create resnext model
model = ResNeXt(img_dim, depth=depth, cardinality=cardinality, width=width, weights=None, classes=n_classes)
print("model created")

model.summary()

optimizer = Adam(lr=1e-3)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
print("finished compiling")

# load dataset iterators
df = pd.read_csv('../csv/train_val_set_earlydr_spectra.csv')
X = df['id'].values
y = df['class'].apply(lambda c: class_map[c]).values
labels = dict(zip(X, y))
del df

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
train_idx, val_idx = next(sss.split(X,y))

X_train, X_val = X[train_idx], X[val_idx]
print('train size', len(X_train))
print('val size', len(X_val))

train_generator = DataGenerator(X_train, labels, **params)
val_generator = DataGenerator(X_val, labels, **params)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if os.path.exists(weights_file):
    model.load_weights(weights_file)
    print("model loaded.")

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
y_pred = model.predict_generator(val_generator)
y_pred = np.argmax(y_pred, axis=1)

# compute accuracy
accuracy = metrics.accuracy_score(y, y_pred) * 100
error = 100 - accuracy
print("accuracy : ", accuracy)
print("error : ", error)
