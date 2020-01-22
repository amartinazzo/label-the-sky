import os,sys,inspect
from glob import glob
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import to_categorical
import numpy as np
import pandas as pd
from models import top_layer_net
import sklearn.metrics as metrics


###

n_epoch = 200

csv_dataset = 'csv/dr1_classes_mag1418_split_ndet.csv'
features_file = 'npy/features_avgpool_depth11_card4_eph500_regression_mag1418.npy'
save_file = f'classifiers/feature-models/eph{n_epoch}_mag1418.h5'
weights_file = save_file

###

n_classes = 3
class_weights = {0: 1, 1: 1.25, 2: 10}
class_map = {'GALAXY': 0, 'STAR': 1, 'QSO': 2}
loss = 'categorical_crossentropy'
metrics_train = ['accuracy']

df = pd.read_csv(csv_dataset)
df = df[(df.split!='test') & (df.n_det==12) & (~df['class'].isna())]

idx_train = df.split=='train'
idx_val = df.split=='val'

X = np.load(features_file)
X_train = X[idx_train]
X_val = X[idx_val]
print('X_train shape', X_train.shape)
print('X_val shape', X_val.shape)

y_train = df.loc[idx_train, 'class'].apply(lambda c: class_map[c])
y_val = df.loc[idx_val, 'class'].apply(lambda c: class_map[c])
y_train = to_categorical(y_train, n_classes)
y_val = to_categorical(y_val, n_classes)
print('y_train shape', y_train.shape)
print('y_val shape', y_val.shape)

model = top_layer_net()
model.summary()

# train
if not os.path.exists(weights_file):
	model.compile(loss=loss, optimizer='adam', metrics=metrics_train)

	callbacks = [
	    ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=10, min_lr=1e-6),
	    ModelCheckpoint(save_file, monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min'),
	    EarlyStopping(monitor='val_loss', patience=20)
	]

	history = model.fit(X_train, y_train,
	    validation_data=(X_val, y_val),
	    epochs=n_epoch,
	    callbacks=callbacks,
	    class_weight=class_weights,
	    verbose=2)

	print('loss', history.history['loss'])
	print('val_loss', history.history['val_loss'])

# make inferences on model
print('predicting')
model.load_weights(save_file)
y_pred = model.predict(X_val)

print(y_pred.shape)

y_pred = np.argmax(y_pred, axis=1)
y_val = np.argmax(y_val, axis=1)
# preds_correct = y_pred==y_val
# x_miss = X_val[~preds_correct]
# print('missclasified', len(x_miss))

# compute accuracy
accuracy = metrics.accuracy_score(y_val, y_pred) * 100
print('accuracy : ', accuracy)

# compute confusion matrix
cm = metrics.confusion_matrix(y_val, y_pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('confusion matrix')
print(cm)