from models import dense_net
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import os


mode = 'train' # train or eval
mag_thres = None #[16, 19]
width = 512


mag_str = 'all-mags' if mag_thres is None else 'mag{}-{}'.format(mag_thres[0], mag_thres[1])
save_file = 'classifiers/catalog-models/dense_{}_{}.h5'.format(mag_str, width)
weights_file = None

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'

class_map = {'GALAXY': 0, 'STAR': 1, 'QSO': 2}

mag_cols = ['u','f378','f395','f410','f430','g','f515','r','f660','i','f861','z']
print('magnitude interval', mag_thres)


def load_dataset(csv_file):
    df = pd.read_csv(csv_file)
    if mag_thres is not None:
        df = df[df.r.between(mag_thres[0], mag_thres[1])]
    print(df['class'].value_counts(normalize=True))
    X = df[mag_cols].values
    y = df['class'].map(lambda s: class_map[s]).values
    y = to_categorical(y, num_classes=len(class_map))

    return X, y


# load datasets

X_train, y_train = load_dataset('csv/matched_cat_dr1_train.csv')
X_val, y_val = load_dataset('csv/matched_cat_dr1_val.csv')

print('train size', len(X_train))
print('val size', len(X_val))

# create model

model = dense_net(input_shape=(len(mag_cols),), width=width)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
if weights_file is not None:
    model.load_weights(weights_file)

# train

if mode=='train':
    callbacks_list = [
        ModelCheckpoint(filepath=save_file, monitor='val_loss', save_best_only=True),
        # EarlyStopping(monitor='val_loss', patience=10)
    ]

    model.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_val, y_val),
    	epochs=100,
        callbacks=callbacks_list,
        verbose=2)

# evaluate

print('predicting')
model.load_weights(save_file)
y_pred = model.predict(X_val, verbose=2)

y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)
print(y_true.shape, y_pred.shape)

cm = confusion_matrix(y_true, y_pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

print('confusion matrix')
print(cm)