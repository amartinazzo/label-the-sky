import os,sys,inspect
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))
from datagen import DataGenerator
from glob import glob
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import pandas as pd
from models import resnext
import sklearn.metrics as metrics
import utils


'''
NOTES

batch of 128 float32 images:
128 * 32 * (32*32*12)/8.4e6 ~= 6 MiB

class proportions:

full set
    0 GALAXY    0.5
    1 STAR      0.4
    2 QSO       0.1

subset with r in (14,18)
    0 GALAXY    0.5
    1 STAR      0.45
    2 QSO       0.05
'''


#########################
# BEGIN PARAMETER SETUP #
#########################

mode = 'train' # train, eval-mags, predict
task = 'regression' # classification or regression (magnitudes)
csv_dataset = 'csv/dr1_classes_split.csv'

n_epoch = 500
img_dim = (32,32,3)
batch_size = 128 #256
cardinality = 4
width = 16
depth = 11 #29

save_file = 'classifiers/image-models/{}_{}bands.h5'.format(task, img_dim[2])
weights_file = None

#######################
# END PARAMETER SETUP #
#######################


n_classes = 3 if task=='classification' else 12
class_weights = {0: 1, 1: 1.3, 2: 5} if task=='classification' else None # normalized 1/class_proportion
data_mode = 'classes' if task=='classification' else 'magnitudes'
extension = 'npy' if img_dim[2]>3 else 'png'
images_folder = '/crops_asinh/' if img_dim[2]>3 else '/crops32/'
lst_activation = 'softmax' if task=='classification' else 'linear'
loss = 'categorical_crossentropy' if task=='classification' else 'mean_squared_error'
metrics_train = ['accuracy'] if task=='classification' else ['mae']


print('csv_dataset', csv_dataset)
print('task', task)
print('images folder', images_folder)
print('batch_size', batch_size)
print('cardinality', cardinality)
print('depth', depth)
print('width', width)
print('nr epochs', n_epoch)
print('save_file', save_file)


models_dir = 'classifiers/image-models/'
data_dir = os.environ['DATA_PATH']+images_folder
params = {'data_folder': data_dir, 'dim': img_dim, 'extension': extension, 'mode': data_mode,'n_classes': n_classes}

# make only 1 gpu visible
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'

# create resnext model
model = resnext(
    img_dim, depth=depth, cardinality=cardinality, width=width, classes=n_classes, last_activation=lst_activation)
print('model created')

model.summary()

model.compile(loss=loss, optimizer='adam', metrics=metrics_train)
print('finished compiling')

# load dataset iterators
df = pd.read_csv(csv_dataset)
# df = df[df.n_det==12]
imgfiles = glob(data_dir+'*/*.'+extension)
imgfiles = [i.split('/')[-1][:-4] for i in imgfiles]
print('original df', df.shape)
df = df[df.id.isin(imgfiles)]
print('df after matching to crops', df.shape)
df_train = df[df.split=='train']
df_val = df[df.split=='val']
del df

X_train, _, labels_train = utils.get_sets(df_train, mode=data_mode)
X_val, y_true, labels_val = utils.get_sets(df_val, mode=data_mode)
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
        ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), patience=10, verbose=1),
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
        class_weight=class_weights,
        verbose=2)

    print('loss', history.history['loss'])
    print('val_loss', history.history['val_loss'])

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
        print('accuracy : ', accuracy)

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


# make inferences on model
print('predicting')
model.load_weights(save_file)
pred_generator = DataGenerator(X_val, shuffle=False, batch_size=1, **params)
y_pred = model.predict_generator(pred_generator, steps=len(y_true))

print(y_true[:5])
print(y_pred[:5])

if task=='classification':
    y_pred = np.argmax(y_pred, axis=1)
    preds_correct = y_pred==y_true
    print('missclasified', X_val[~preds_correct].shape)

    # compute accuracy
    accuracy = metrics.accuracy_score(y_true, y_pred) * 100
    print('accuracy : ', accuracy)

    # compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('confusion matrix')
    print(cm)

else:
    # y_true = 30*y_true
    # y_pred = 30*y_pred
    print('y_true\n', y_true[:5])
    print('y_pred\n', y_pred[:5])
    abs_errors = np.absolute(y_true - y_pred)
    print('absolute errors\n', abs_errors[:5])
    print('\n\nMAE:', np.mean(abs_errors))
    print('MAPE:', np.mean(abs_errors / y_true) * 100)
