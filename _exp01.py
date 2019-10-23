from datagen import DataGenerator
from datetime import date
from glob import glob
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input
from keras.layers.core import  Dense
from keras.models import Model
import numpy as np
import pandas as pd
import pickle
from classifiers.models import ResNeXt29
import os
import sklearn.metrics as metrics
import sys
from time import time
from umap import UMAP
from utils import get_sets


def get_class_weights(csv_file):
    df = pd.read_csv(csv_file)
    df = df[(df.photoflag==0)&(df.ndet==12)]
    print('dataset size', df.shape)
    return np.round(1/df['class'].value_counts(normalize=True).values, 1)


def build_dataset(csv_file, data_folder, input_dim, n_outputs, target, split=None, batch_size=32):
    df = pd.read_csv(csv_file)
    df = df[(df.photoflag==0)&(df.ndet==12)]
    class_weights = np.round(1/df['class'].value_counts(normalize=True).values, 1)

    if split is not None:
        df = df[df.split==split]
    else:
        split = 'full'

    n_bands = input_dim[-1]
    X, y, labels = get_sets(df, target=target, n_bands=n_bands)
    print(f'{split} size', len(X))

    params = {
        'batch_size': batch_size,
        'data_folder': data_folder,
        'input_dim': input_dim,
        'n_outputs': n_outputs,
        'target': target
        }

    data_gen = DataGenerator(X, labels=labels, **params)

    return X, y, data_gen


def build_model(input_dim, n_outputs, lst_activation, loss, metrics, output_feature_dim=512,
    top_layer=True, weights_file=None, width=64, card=4):
    print('width', width)
    print('cardinality', card)

    model = ResNeXt29(
        input_dim, num_classes=n_outputs, width=width, cardinality=card,
        last_activation=lst_activation, output_dim=output_feature_dim)

    if weights_file is not None and os.path.exists(weights_file):
        model.load_weights(weights_file)
        print('loaded weights')

    if not top_layer:
        model.layers.pop()

    model.compile(loss=loss, optimizer='adam', metrics=metrics)

    # trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    # non_trainable_count = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
    # print('total params: {:,}'.format(trainable_count + non_trainable_count))
    # print('trainable params: {:,}'.format(trainable_count))
    # print('non-trainable params: {:,}'.format(non_trainable_count))

    return model


def train(model, train_gen, val_gen, model_file, class_weights=None, epochs=500):
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1),
        ModelCheckpoint(model_file, monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min'),
        EarlyStopping(monitor='val_loss', mode='min', patience=8, restore_best_weights=True, verbose=1)
    ]

    history = model.fit_generator(
        generator=train_gen,
        validation_data=val_gen,
        steps_per_epoch=len(train_gen),
        validation_steps=len(val_gen),
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=2)

    return history


def build_classifier(input_dim, n_classes=3):
    inputs = Input(shape=(input_dim,))
    x = Dense(512, activation='relu')(inputs)
    outputs = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


def train_classifier(model, X_train, y_train, X_val, y_val, clf_file, class_weights=None, batch_size=32, epochs=10):
    callbacks = [
        ModelCheckpoint(clf_file, monitor='val_accuracy', save_best_only=True, save_weights_only=True, mode='max'),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        epochs=epochs,
        verbose=2)

    return history


def compute_error(y_pred, y_true, target):
    if target=='classes':
        y_pred_arg = np.argmax(y_pred, axis=1)
        y_true_arg = np.argmax(y_true, axis=1)
        accuracy = metrics.accuracy_score(y_true_arg, y_pred_arg)
        print('accuracy:', accuracy)
        cm = metrics.confusion_matrix(y_true_arg, y_pred_arg)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('confusion matrix')
        print(cm)
    else:
        err_abs = np.absolute(y_true-y_pred)
        df = pd.DataFrame(err_abs)
        print(df.describe().to_string())
        print('MAE:', np.mean(err_abs))
        print('MAPE:', np.mean(err_abs/y_true)*100)


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
    'magnitudes': 'relu',
    'redshifts': 'sigmoid',
}

loss_switch = {
    'classes': 'categorical_crossentropy',
    'magnitudes': 'mean_absolute_error',
    'redshifts': 'mean_absolute_error',
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
    * feature vector dim (512, 256, 128, 64)
    * nbands (12, 5, 3)
    * target (classes, magnitudes, redshifts)
    (ordered from outer to inner loop)

    total runs: 4*3*3 = 36

    '''

    if len(sys.argv) != 6:
        print('usage: python %s <data_dir> <csv_file> <target> <nbands> <feature_vector_dim>' % sys.argv[0])
        exit(1)

    # read input args
    data_dir = sys.argv[-5] #os.getenv('HOME')+'/label_the_sky'
    csv_file = sys.argv[-4]
    target = sys.argv[-3]
    n_bands = int(sys.argv[-2])
    output_dim = int(sys.argv[-1])

    print('data_dir', data_dir)
    print('csv_file', csv_file)
    print('target', target)
    print('n_bands', n_bands)
    print('output_dim', output_dim)

    # set parameters
    n_outputs = n_outputs_switch.get(target+str(n_bands))
    extension = extension_switch.get(n_bands)
    images_folder = os.path.join(data_dir, images_folder_switch.get(n_bands))
    lst_activation = last_activation_switch.get(target)
    loss = loss_switch.get(target)
    metrics_train = metrics_switch.get(target)

    input_dim = (32, 32, n_bands)
    # model_name = '{}_{}_{}bands_{}'.format(date.today().strftime('%y%m%d'), target, n_bands, output_dim)
    # clf_name = '{}_{}_{}bands_{}'.format(date.today().strftime('%y%m%d'), 'topclf', n_bands, output_dim)
    model_name = '{}_{}_{}bands_{}'.format(191018, target, n_bands, output_dim)
    clf_name = '{}_{}_{}bands_{}'.format(191018, 'topclf', n_bands, output_dim)
    model_file = data_dir+f'/trained_models/{model_name}.h5'
    clf_file = data_dir+f'/trained_models/{clf_name}.h5'
    results_folder = os.getenv('HOME')+'/label_the_sky/results'
    print('results_folder', results_folder)

    start = time()

    # print('training backbone')
    class_weights = get_class_weights(csv_file)
    X_train, y_train, train_gen = build_dataset(csv_file, images_folder, input_dim, n_outputs, target, 'train')
    X_val, y_val, val_gen = build_dataset(csv_file, images_folder, input_dim, n_outputs, target, 'val')
    model = build_model(input_dim, n_outputs, lst_activation, loss, metrics_train, output_dim)

    # history = train(model, train_gen, val_gen, model_file, class_weights)
    # with open(os.path.join(results_folder, f'{model_name}_history.pkl'), 'wb') as f:
    #     pickle.dump(history.history, f)
    # print('--- minutes taken:', int((time()-start)/60))

    print('evaluating model')
    train_gen = DataGenerator(
        X_train, shuffle=False, batch_size=1, data_folder=images_folder, input_dim=input_dim, n_outputs=n_outputs, target=target)
    val_gen = DataGenerator(
        X_val, shuffle=False, batch_size=1, data_folder=images_folder, input_dim=input_dim, n_outputs=n_outputs, target=target)
    model = build_model(input_dim, n_outputs, lst_activation, loss, metrics_train, output_dim, weights_file=model_file)
    y_val_hat, X_val_feats = model.predict_generator(val_gen)
    compute_error(y_val_hat, y_val, target)
    np.save(os.path.join(results_folder, f'{model_name}_y_train.npy'), y_train)
    np.save(os.path.join(results_folder, f'{model_name}_y_val.npy'), y_val)
    np.save(os.path.join(results_folder, f'{model_name}_y_val_hat.npy'), y_val_hat)
    print('--- minutes taken:', int((time()-start)/60))

    if model_name != '191018_classes_12bands_1024':
        print('extracting features')
        train_gen = DataGenerator(
            X_train, shuffle=False, batch_size=1, data_folder=images_folder, input_dim=input_dim, n_outputs=n_outputs, target=target)
        model = build_model(input_dim, n_outputs, lst_activation, loss, metrics_train, output_dim, weights_file=model_file)
        y_train_hat, X_train_feats = model.predict_generator(train_gen)
        X_val_feats = model.predict_generator(val_gen)
        np.save(os.path.join(results_folder, f'{model_name}_X_train_features.npy'), X_train_feats)
        np.save(os.path.join(results_folder, f'{model_name}_X_val_features.npy'), X_val_feats)
        print('--- minutes taken:', int((time()-start)/60))

    print('training dense classifier')        
    clf = build_classifier(X_train_feats.shape[1])
    clf_history = train_classifier(clf, X_train_feats, y_train, X_val_feats, y_val, clf_file, class_weights)
    y_feats_pred = clf.predict(X_val_feats)
    compute_error(y_feats_pred, y_val, 'classes')
    print('--- minutes taken:', int((time()-start)/60))

    print('extracting UMAP projections')
    X_features = np.concatenate([X_train_feats, X_val_feats])
    X_umap = UMAP().fit_transform(X_features)
    np.save(os.path.join(results_folder, f'{model_name}_X_features_umap.npy'), X_umap)
    print('--- minutes taken:', int((time()-start)/60))