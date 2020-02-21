'''
exp01:

01. build dataset
02. build model
03. train
04. predict
05. compute error
06. extract features
07. train dense classifier
08. compute error
09. generate 2d projections

input args:
* backbone      (resnext, efficientnet, vgg)
* n_bands       (12, 5, 3)
* target        (classes, magnitudes)
(ordered from outer to inner loop)

total runs: 3*3*2 = 18

part II:
vary dataset_perc in [1, 100]
* dataset_perc  (10, 25, 50, 75)

'''


from datagen import DataGenerator
from efficientnet.keras import EfficientNetB0
from glob import glob
import json
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input
from keras.layers import  Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Model
from keras.optimizers import SGD
from keras_lookahead import Lookahead
from keras_radam import RAdam
from models.callbacks import TimeHistory
from models.resnext import ResNeXt29
from models.vgg import VGG11b, VGG16
import numpy as np
import pandas as pd
import os
from sklearn.metrics import classification_report, confusion_matrix
import sys
import tensorflow as tf
from time import time
from umap import UMAP
from utils import get_sets
import warnings


def set_random_seeds():
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)
    tf.set_random_seed(420)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)


def get_class_weights(df):
    x = 1/df['class'].value_counts(normalize=True).values
    x = np.round(x / np.max(x), 4)
    return x


def build_dataset(df, data_folder, input_dim, n_outputs, target, split=None, batch_size=32):
    if split is not None:
        df = df[df.split==split]
    else:
        split = 'full'

    n_bands = input_dim[-1]
    ids, y, labels = get_sets(df, target=target, n_bands=n_bands)
    print(f'{split} size', len(ids))

    shuffle = False if split!='train' else True
    bs = 1 if split=='test' else batch_size

    params = {
        'batch_size': bs,
        'data_folder': data_folder,
        'input_dim': input_dim,
        'n_outputs': n_outputs,
        'target': target,
        'shuffle': shuffle,
        }

    data_gen = DataGenerator(ids, labels=labels, **params)

    return ids, y, data_gen


def build_model(
    input_dim, n_outputs, last_activation, loss, backbone='resnext', include_top=True, include_features=False,
    weights_file=None, metrics=['accuracy']):
    if backbone=='resnext':
        model = ResNeXt29(
            input_shape=input_dim, num_classes=n_outputs, last_activation=last_activation,
            include_top=include_top, include_features=include_features)
    elif backbone=='efficientnet':
        model = EfficientNetB0(
            input_shape=input_dim, classes=n_outputs, last_activation=last_activation,
            include_top=include_top, include_features=include_features, weights=None)
    elif backbone=='vgg16':
        model = VGG16(
            input_shape=input_dim, num_classes=n_outputs, last_activation=last_activation,
            include_top=include_top, include_features=include_features)
    elif backbone=='vgg11':
        model = VGG11b(
            input_shape=input_dim, num_classes=n_outputs, last_activation=last_activation,
            include_top=include_top, include_features=include_features)
    else:
        print('accepted backbones: resnext, efficientnet, vgg16, vgg11')
        exit()

    if weights_file is not None and os.path.exists(weights_file):
        model.load_weights(weights_file)
        print('loaded weights')

    if n_outputs==3:
        lr = 0.01
    else:
        lr = 0.001

    optimizer = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True) #RAdam() #Lookahead(RAdam())
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable_count = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
    print('total params: {:,}'.format(trainable_count + non_trainable_count))
    print('trainable params: {:,}'.format(trainable_count))
    print('non-trainable params: {:,}'.format(non_trainable_count))

    return model


def train(model, gen_train, gen_val, model_file, class_weights=None, epochs=500, verbose=True):
    time_callback = TimeHistory()
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1),
        ModelCheckpoint(model_file, monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min'),
        EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1),
        time_callback
    ]

    history = model.fit_generator(
        generator=gen_train,
        validation_data=gen_val,
        steps_per_epoch=len(gen_train),
        validation_steps=len(gen_val),
        epochs=1,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=2)

    if verbose:
        print('History')
        print(history.history)
        print('Time taken per epoch (s)')
        print(time_callback.times)

    return history


def build_classifier(input_dim, n_intermed=12, n_classes=3, layer_type='dense'):
    if type(input_dim)==int:
        input_dim = (input_dim,)
    inputs = Input(shape=input_dim)
    if layer_type=='conv':
        x = Conv2D(32, kernel_size=3, activation='relu')(inputs)
        x = MaxPooling2D(pool_size=7, stride=7)(x)
        x = Flatten()(x)
        x = Dense(n_intermed, activation='relu')(x)
    else:
        x = Dense(n_intermed, activation='relu')(inputs)
    x = Dropout(0.5)(x)
    outputs = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs, outputs)

    optimizer = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True) #RAdam() #Lookahead(RAdam())
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()

    return model


def train_classifier(model, X_train, y_train, X_val, y_val, clf_file, class_weights=None, batch_size=32, epochs=300, verbose=True):
    time_callback = TimeHistory()
    callbacks = [
        ModelCheckpoint(clf_file, monitor='val_accuracy', save_best_only=True, save_weights_only=True, mode='max'),
        EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1),
        time_callback,
    ]

    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        epochs=epochs,
        validation_data=(X_val, y_val),
        verbose=2)

    if verbose:
        print('History')
        print(history.history)
        print('Time taken per epoch (s)')
        print(time_callback.times)

    return history


def compute_metrics(y_pred, y_true, target='classes', onehot=True):
    if target=='classes':
        if onehot:
            y_pred_arg = np.argmax(y_pred, axis=1)
            y_true_arg = np.argmax(y_true, axis=1)
        else:
            y_pred_arg = np.copy(y_pred)
            y_true_arg = np.copy(y_true)
        print(classification_report(y_true_arg, y_pred_arg, target_names=['GALAXY', 'STAR', 'QSO']))
        cm = confusion_matrix(y_true_arg, y_pred_arg)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm, 2)
        print('confusion matrix')
        print(cm)

    else:
        err_abs = np.absolute(y_true-y_pred)
        df = pd.DataFrame(err_abs)
        print(df.describe().to_string())
        df = pd.DataFrame(err_abs*30)
        print(df.describe().to_string())
        err_abs = 30*err_abs
        print('MAE:', np.mean(err_abs))
        print('MAPE:', np.mean(err_abs/(30*y_true))*100)


def relu_saturated(x):
    return K.relu(x, max_value=1.)

###########
# SWITCHS #
###########


n_outputs_switch = {
    'classes12': 3,
    'classes5': 3,
    'classes3': 3,
    'magnitudes12': 12,
    'magnitudes5': 5,
    'magnitudes3': 12,
    'redshifts12': 2,
    'redshifts5': 2,
    'redshifts3': 2,
}

extension_switch = {
    3: '.png',
    5: '.npy',
    12: '.npy'
}

images_folder_switch = {
    3: 'crops_rgb32',
    5: 'crops_calib',
    12: 'crops_calib'
}

last_activation_switch = {
    'classes': 'softmax',
    'magnitudes': relu_saturated, #'relu',
    'redshifts': 'sigmoid',
}

loss_switch = {
    'classes': 'categorical_crossentropy',
    'magnitudes': 'mae',
    'redshifts': 'mae',
}

metrics_switch = {
    'classes': ['accuracy'],
    'magnitudes': None,
    'redshifts': None,
}


########
# MAIN #
########


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    set_random_seeds()

    if len(sys.argv) < 7:
        print('usage: python %s <data_dir> <csv_file> <backbone> <target> <nbands> <timestamp> <dataset_perc : optional>' % sys.argv[0])
        exit(1)

    # read input args
    data_dir = sys.argv[1]
    csv_file = sys.argv[2]
    backbone = sys.argv[3]
    target = sys.argv[4]
    n_bands = int(sys.argv[5])
    timestamp = sys.argv[6]
    dataset_perc = int(sys.argv[7])/100. if len(sys.argv) == 8 else 1.

    print('data_dir', data_dir)
    print('csv_file', csv_file)
    print('backbone', backbone)
    print('target', target)
    print('n_bands', n_bands)
    print('dataset_perc', dataset_perc)

    # set parameters
    n_outputs = n_outputs_switch.get(target+str(n_bands))
    extension = extension_switch.get(n_bands)
    images_folder = os.path.join(data_dir, images_folder_switch.get(n_bands))
    lst_activation = last_activation_switch.get(target)
    loss = loss_switch.get(target)
    metrics = metrics_switch.get(target)

    print('n_outputs', n_outputs)
    print('images_folder', images_folder)
    print('last activation', lst_activation)
    print('loss', loss)
    print('metrics', metrics)

    input_dim = (32, 32, n_bands)
    model_name = '{}_{}_{}_{}'.format(timestamp, backbone, target, n_bands)
    clf_name = '{}_{}_{}_{}_topclf'.format(timestamp, backbone, target, n_bands)
    model_file = data_dir+f'/trained_models/{model_name}.h5'
    clf_file = data_dir+f'/trained_models/{clf_name}.h5'
    results_folder = os.getenv('HOME')+'/label_the_sky/results'
    print('results_folder', results_folder)

    start = time()

    df = pd.read_csv(csv_file)
    orig_shape = df.shape

    if target=='magnitudes':
        e = 0.5
        df = df[(
            df.u_err <= e) & (df.f378_err <= e) & (df.f395_err <= e) & (df.f410_err <= e) & (df.f430_err <= e) & (df.g_err <= e) & (
            df.f515_err <= e) & (df.r_err <= e) & (df.f660_err <= e) & (df.i_err <= e) & (df.f861_err <= e) & (df.z_err <= e)]

    if dataset_perc < 1:
        print('generating subset of data')
        df['random'] = np.random.rand(df.shape[0])
        df = df[df.random <= dataset_perc]

    print('new shape', df.shape)
    print('proportion', df.shape[0]/orig_shape[0])
    print('split proportions')
    print(df.split.value_counts(normalize=True))
    print('class proportions')
    print(df['class'].value_counts(normalize=True))
    class_weights = get_class_weights(df)
    print('class weights', class_weights)

    print('training backbone')
    X_train, y_train, gen_train = build_dataset(df, images_folder, input_dim, n_outputs, target, 'train')
    X_val, y_val, gen_val = build_dataset(df, images_folder, input_dim, n_outputs, target, 'val')
    X_test, y_test, gen_test = build_dataset(df, images_folder, input_dim, n_outputs, target, 'test')

    model = build_model(input_dim, n_outputs, lst_activation, loss, backbone, metrics=metrics)
    history = train(model, gen_train, gen_val, model_file, class_weights)
    # json.dump(dict(history.history), open(f'history/history_{model_name}.json', 'w'))
    print('--- minutes taken:', int((time()-start)/60))

    print('evaluating model')
    model = build_model(
        input_dim, n_outputs, lst_activation, loss, backbone,
        include_top=True, include_features=True, weights_file=model_file)
    y_test_hat, X_test_feats = model.predict_generator(gen_test)
    compute_metrics(y_test, y_test_hat, target)
    np.save(os.path.join(results_folder, f'{model_name}_y_test.npy'), y_test)
    np.save(os.path.join(results_folder, f'{model_name}_y_test_hat.npy'), y_test_hat)
    np.save(os.path.join(results_folder, f'{model_name}_X_test_features.npy'), X_test_feats)
    print('--- minutes taken:', int((time()-start)/60))

    if target!='classes':
        print('training dense classifier')
        gen_train = DataGenerator(
            X_train, shuffle=False, batch_size=1, data_folder=images_folder, input_dim=input_dim, n_outputs=n_outputs, target=target)
        gen_val = DataGenerator(
            X_val, shuffle=False, batch_size=1, data_folder=images_folder, input_dim=input_dim, n_outputs=n_outputs, target=target)
        y_train_hat, X_train_feats = model.predict_generator(gen_train)
        y_val_hat, X_val_feats = model.predict_generator(gen_val)
        np.save(os.path.join(results_folder, f'{model_name}_X_train_features.npy'), X_train_feats)
        np.save(os.path.join(results_folder, f'{model_name}_X_val_features.npy'), X_val_feats)

        _, y_train, _ = build_dataset(df, images_folder, input_dim, n_outputs, 'classes', 'train')
        _, y_val, _ = build_dataset(df, images_folder, input_dim, n_outputs, 'classes', 'val')
        _, y_test, _ = build_dataset(df, images_folder, input_dim, n_outputs, 'classes', 'test')
        
        clf = build_classifier(X_train_feats.shape[1])
        clf.load_weights(clf_file)
        clf_history = train_classifier(clf, X_train_feats, y_train, X_val_feats, y_val, clf_file, class_weights)
        y_test_feats_hat = clf.predict(X_test_feats)
        compute_metrics(y_test, y_test_feats_hat, 'classes')
        print('--- minutes taken:', int((time()-start)/60))

    print('extracting UMAP projections')
    X_features = np.copy(X_test_feats)
    X_umap = UMAP().fit(X_features).embedding_
    np.save(os.path.join(results_folder, f'{model_name}_X_test_features_umap.npy'), X_umap)
    print('--- minutes taken:', int((time()-start)/60))
