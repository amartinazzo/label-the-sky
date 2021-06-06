"""
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
* target        (classes, magnitudes, magnitudesmock)
(ordered from outer to inner loop)

total runs: 3*3*2 = 18

"""


from datagen import DataGenerator
from efficientnet.tfkeras import EfficientNetB0
import json
from tensorflow.compat.v1 import disable_eager_execution
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from models.callbacks import TimeHistory
from models.resnext import ResNeXt29
from models.vgg import VGG11b, VGG16
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys
import tensorflow as tf
from time import time
from umap import UMAP
from utils import get_sets
import warnings


def make_serializable(hist):
    d = {}
    for k in hist.keys():
        d[k] = [float(item) for item in hist[k]]
    return d

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
    'magnitudesmock12': 12,
    'magnitudesmock5': 5,
    'magnitudesmock3': 12,
    'redshifts12': 2,
    'redshifts5': 2,
    'redshifts3': 2,
}

images_folder_switch = {
    3: 'crops_rgb',
    5: 'crops_calib',
    12: 'crops_calib'
}

last_activation_switch = {
    'classes': 'softmax',
    'magnitudes': relu_saturated,  # 'relu',
    'magnitudesmock': relu_saturated,  # 'relu',
    'redshifts': 'sigmoid',
}

loss_switch = {
    'classes': 'categorical_crossentropy',
    'magnitudes': 'mae',
    'magnitudesmock': 'mae',
    'redshifts': 'mae',
}

metrics_switch = {
    'classes': ['accuracy'],
    'magnitudes': None,
    'magnitudesmock': None,
    'redshifts': None,
}


########
# MAIN #
########


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    set_random_seeds()

    if len(sys.argv) < 5:
        print('usage: python {} <backbone> <target> <nbands> <timestamp>'.format(
            sys.argv[0]))
        exit(1)

    # read input args
    backbone = sys.argv[1]
    target = sys.argv[2]
    n_bands = int(sys.argv[3])
    timestamp = sys.argv[4]

    # data_dir = os.environ['DATA_PATH']
    data_dir = os.environ['HOME']
    base_dir = os.environ['HOME'] + '/mnt/label-the-sky'
    npy_folder = base_dir + '/npy'

    csv_file = base_dir + '/csv/dr1_split.csv'

    print('data_dir', data_dir)
    print('csv_file', csv_file)
    print('backbone', backbone)
    print('target', target)
    print('n_bands', n_bands)

    # set parameters
    n_outputs = n_outputs_switch.get(target + str(n_bands))
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
    clf_name = '{}_{}_{}_{}_clf'.format(timestamp, backbone, target, n_bands)
    model_file = data_dir + f'/trained_models/{model_name}.h5'
    clf_file = data_dir + f'/trained_models/{clf_name}.h5'

    start = time()

    disable_eager_execution()

    df = pd.read_csv(csv_file)
    df = df[df.pretraining]

    print('split proportions')
    print(df.split.value_counts(normalize=True))
    print('class proportions')
    print(df['class'].value_counts(normalize=True))
    class_weights = get_class_weights(df) if target == 'classes' else None
    print('class weights', class_weights)

    # print('training backbone')
    # X_train, y_train, gen_train = build_dataset(
    #     df, images_folder, input_dim, n_outputs, target, 'train')
    # X_val, y_val, gen_val = build_dataset(
    #     df, images_folder, input_dim, n_outputs, target, 'val')
    X_test, y_test, gen_test = build_dataset(
        df, images_folder, input_dim, n_outputs, target, 'test')

    # model = build_backbone(
    #     input_dim, n_outputs, lst_activation, loss, backbone, metrics=metrics)
    # history = train_backbone(
    #     model, gen_train, gen_val, model_file, class_weights)
    # with open(f'history/history_{model_name}.json', 'w') as f:
    #     json.dump(make_serializable(history.history), f)
    # print('--- minutes taken:', int((time() - start) / 60))

    print('evaluating model')
    model = build_backbone(
        input_dim, n_outputs, lst_activation, loss, backbone,
        include_top=True, include_features=True, weights_file=model_file)
    y_test_hat, X_test_feats = model.predict_generator(gen_test)
    compute_metrics(y_test, y_test_hat, target)

    # np.save(os.path.join(npy_folder, f'{model_name}_y_test.npy'), y_test)
    # np.save(os.path.join(
    #     npy_folder, f'{model_name}_y_test_hat.npy'), y_test_hat)
    # np.save(os.path.join(
    #     npy_folder, f'{model_name}_X_test_features.npy'), X_test_feats)
    # print('--- minutes taken:', int((time() - start) / 60))

    print('training dense classifier')

    df_clf = pd.read_csv(csv_file)
    df_clf = df_clf[~df_clf.pretraining]

    # compute validation and testing feature sets (fixed)

    _, _, gen_val = build_dataset(
        df_clf, images_folder, input_dim, n_outputs, target,
        'val', bs=1, shuffle=False)
    _, _, gen_test = build_dataset(
        df_clf, images_folder, input_dim, n_outputs, target, 'test')

    y_val_hat, X_val_feats = model.predict_generator(gen_val)
    y_test_hat, X_test_feats = model.predict_generator(gen_test)

    _, y_val, _ = build_dataset(
        df_clf, images_folder, input_dim, 3, 'classes', 'val')
    _, y_test, _ = build_dataset(
        df_clf, images_folder, input_dim, 3, 'classes', 'test')

    # vary training set size

    acc = []
    # for p in np.linspace(0.05, 1, 20):
    df_clf_train = df_clf #[df_clf.random <= p]
    # print('percentage of full train', df_clf_train[
    #     df_clf_train.split == 'train'].shape[0]/df_clf[
    #     df_clf.split == 'train'].shape[0])
    class_weights = get_class_weights(df_clf_train)
    print('class weights', class_weights)

    _, _, gen_train = build_dataset(
        df_clf_train, images_folder, input_dim, n_outputs, target,
        'train', bs=1, shuffle=False)
    y_train_hat, X_train_feats = model.predict_generator(gen_train)

    _, y_train, _ = build_dataset(
        df_clf_train, images_folder, input_dim, 3, 'classes', 'train')

    clf = build_classifier(X_train_feats.shape[1])
    clf_history = train_classifier(
        clf, X_train_feats, y_train, X_val_feats, y_val,
        clf_file, class_weights)

    # with open(f'history/history_{clf_name}_p{p}.json', 'w') as f:
    with open(f'history/history_{clf_name}.json', 'w') as f:
        json.dump(make_serializable(clf_history.history), f)
    y_test_feats_hat = clf.predict(X_test_feats)
    compute_metrics(y_test, y_test_feats_hat, 'classes')
    acc.append(
        accuracy_score(
            np.argmax(y_test, axis=1),
            np.argmax(y_test_feats_hat, axis=1)))
    print('--- minutes taken:', int((time() - start) / 60))

    print('accuracies', acc)

    print('extracting UMAP projections')
    X_features = np.copy(X_test_feats)
    X_umap = UMAP().fit(X_features).embedding_
    np.save(os.path.join(
        npy_folder, f'{model_name}_X_test_features_umap.npy'), X_umap)
    print('--- minutes taken:', int((time() - start) / 60))
