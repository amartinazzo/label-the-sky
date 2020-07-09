import efficientnet
from efficientnet.tfkeras import EfficientNetB2
import json
from keras_applications import vgg16, resnext
from models.callbacks import TimeHistory
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # info and warning messages are not printed

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD


BACKBONES = ['efficientnet', 'resnext', 'vgg', None]
BATCH_SIZE = 32
BROAD_BANDS = [0, 5, 7, 9, 11]
CLASS_NAMES = ['GALAXY', 'STAR', 'QSO']
MAG_MAX = 35.
N_CHANNELS = [3, 5, 12]
N_CLASSES = 3
OUTPUT_TYPES = ['class', 'magnitudes', 'mockedmagnitudes']
SPLITS = ['train', 'val', 'test']

BACKBONE_FN = {
    'efficientnet': EfficientNetB2,
    'resnext': resnext.ResNeXt50,
    'vgg': vgg16.VGG16
}

PREPROCESSING_FN = {
    'efficientnet': efficientnet.model.preprocess_input,
    'resnext': resnext.preprocess_input,
    'vgg': vgg16.preprocess_input
}


def compute_metrics(y_pred, y_true, target='class', onehot=True):
    if target not in OUTPUT_TYPES:
        raise ValueError('target should be one of %s, but %s was given' % (
            OUTPUT_TYPES, target))

    if target == 'class':
        if onehot:
            y_pred_arg = np.argmax(y_pred, axis=1)
            y_true_arg = np.argmax(y_true, axis=1)
        else:
            y_pred_arg = np.copy(y_pred)
            y_true_arg = np.copy(y_true)
        print(y_true.shape)
        print(classification_report(
            y_true_arg, y_pred_arg, target_names=CLASS_NAMES, digits=4))
        cm = confusion_matrix(y_true_arg, y_pred_arg)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm, 2)
        print('confusion matrix')
        print(cm)
    elif target in ['magnitudes', 'mockedmagnitudes']:
        err_abs = np.absolute(y_true-y_pred)
        df = pd.DataFrame(err_abs)
        print(df.describe().to_string())
        df = pd.DataFrame(err_abs*MAG_MAX)
        print(df.describe().to_string())
        err_abs = MAG_MAX*err_abs
        print('MAE:', np.mean(err_abs))
        print('MAPE:', np.mean(err_abs/(MAG_MAX*y_true))*100)


def relu_saturated(x):
    return keras.backend.relu(x, max_value=1.)


def serialize(history):
    d = {}
    for k in history.keys():
        d[k] = [float(item) for item in history[k]]
    return d


def set_random_seeds():
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)
    tf.random.set_seed(420)
    # session_conf = tf.ConfigProto(
    #     intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    # sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    # K.set_session(sess)


class Trainer:
    def __init__(self, backbone, n_channels, output_type, base_dir, weights, model_name):
        if backbone not in BACKBONES:
            raise ValueError('backbone should be one of %s, but %s was given' % (
                BACKBONES, backbone))

        if n_channels not in N_CHANNELS:
            raise ValueError('n_channels should be one of %s, but %s was given' % (
                N_CHANNELS, n_channels))

        if output_type not in OUTPUT_TYPES:
            raise ValueError('output_type should be one of %s, but %s was given' % (
                OUTPUT_TYPES, output_type))

        if weights is not None and weights != 'imagenet' and not os.path.exists(weights):
            raise ValueError('weights must be: None, imagenet, or a valid path to a h5 file')

        self.backbone = backbone
        self.n_channels = n_channels
        self.output_type = output_type
        self.weights = weights
        self.model_name = model_name

        self.input_shape = (32, 32, n_channels)
        self.max_norm = max_norm(2)

        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, 'data')
        self.save_dir = os.path.join(base_dir, 'trained_models')

        if self.output_type == 'class':
            self.activation = 'softmax'
            self.loss = 'categorical_crossentropy'
            self.metrics = ['accuracy']
            self.n_outputs = 3
        else:
            self.activation = relu_saturated
            self.loss = 'mae'
            self.metrics = None
            self.n_outputs = 12 if self.n_channels != 5 else 5

        tf.compat.v1.disable_eager_execution()
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

        self.set_callbacks()
        if self.backbone is not None:
            self.build_model(learning_rate=0.01)

    def load_data(self, split, subset):
        if subset not in ['pretraining', 'clf']:
            raise ValueError('subset must be: pretraining, clf')

        channels = 12 if self.n_channels==5 else self.n_channels

        X = np.load(os.path.join(
            self.data_dir,
            f'{subset}_{channels}_X_{split}.npy'))

        y = np.load(os.path.join(
            self.data_dir,
            f'{subset}_{channels}_y_{self.output_type}_{split}.npy'))

        return X, y

    def load_magnitudes(self, split):
        X = np.load(os.path.join(
            self.data_dir,
            f'clf_12_y_magnitudes_{split}.npy'))

        y = np.load(os.path.join(
            self.data_dir,
            f'clf_12_y_class_{split}.npy'))

        return X, y

    def preprocess_input(self, X):
        if self.n_channels==3 and X.dtype=='uint8':
            preprocessing_fn = PREPROCESSING_FN.get(self.backbone)
            Xp = preprocessing_fn(
                X,
                backend=keras.backend,
                layers=keras.layers,
                models=keras.models,
                utils=keras.utils
            )
        elif self.n_channels==5 and X.shape[-1]>5:
            Xp = X[:, :, :, BROAD_BANDS]
        else:
            Xp = X

        if Xp.dtype != 'float32':
            raise ValueError('Xp data type should be float32')

        return Xp

    def preprocess_output(self, y):
        if y.dtype!='uint8' and self.output_type!='class':
            yp = y / MAG_MAX
        else:
            yp = y
        return yp

    def build_model(self, learning_rate, freeze_backbone=False):
        architecture_fn = BACKBONE_FN.get(self.backbone)

        weights0 = 'imagenet' if self.weights == 'imagenet' else None
        model = architecture_fn(
            input_shape=self.input_shape,
            include_top=False,
            weights=weights0,
            backend=keras.backend,
            layers=keras.layers,
            models=keras.models,
            utils=keras.utils
        )

        x = model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(
            1024,
            activation=LeakyReLU(),
            kernel_initializer='glorot_uniform',
            kernel_constraint=self.max_norm)(x)

        # top layer
        y = Dropout(0.5)(x)
        y = Dense(self.n_outputs, activation=self.activation)(y)

        self.top_layer_idx = -4
        self.model = Model(inputs=model.input, outputs=y)

        self.embedder_x = Model(inputs=self.model.input, outputs=x)
        self.embedder_yx = Model(inputs=self.model.input, outputs=[y, x])

        if self.weights is not None and os.path.exists(self.weights):
            self.model.load_weights(self.weights, by_name=True, skip_mismatch=True)
            print('loaded .h5 weights')

        if freeze_backbone:
            for layer in self.model.layers[:self.top_layer_idx]:
                layer.trainable = False

        opt = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss=self.loss, optimizer=opt, metrics=self.metrics)

    def build_top_clf(self, inpt_dim):
        # TODO
        inpt = Input(shape=(inpt_dim,))
        x = Dense(12, activation=LeakyReLU())(inpt)
        x = Dense(N_CLASSES, activation=self.activation)(x)
        self.clf = Model(inpt, x)

        opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.clf.compile(loss=self.loss, optimizer=opt, metrics=self.metrics)

    def describe(self, verbose=False):
        if verbose:
            self.model.summary()

        if self.model is not None:
            trainable = np.sum([
                keras.backend.count_params(w) for w in self.model.trainable_weights])
            non_trainable = np.sum([
                keras.backend.count_params(w) for w in self.model.non_trainable_weights])
            print('******************************')
            print('total params\t', f'{int(trainable + non_trainable):,}')
            print('trainable\t', f'{int(trainable):,}')
            print('non-trainable\t', f'{int(non_trainable):,}')
            print()

        print('backbone\t', self.backbone)
        print('n_channels\t', self.n_channels)
        print('output\t\t', self.output_type)
        print('weights\t\t', self.weights)
        print('******************************')

    def set_callbacks(self):
        self.callbacks = [
            ReduceLROnPlateau(
                monitor='val_loss', factor=0.1, patience=5, verbose=1),
            ModelCheckpoint(
                os.path.join(self.base_dir, 'trained_models', self.model_name+'.h5'),
                monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min'),
            EarlyStopping(
                monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1),
        ]

    def set_class_weights(self, y):
        if self.output_type != 'class':
            self.class_weights = None
        else:
            if len(y.shape) > 1:
                yy = np.argmax(y, axis=1)
            self.class_weights = compute_class_weight('balanced', np.unique(yy), yy)
            print('set class weights to', self.class_weights)

    def train(self, X_train, y_train, X_val, y_val, epochs=500, runs=3):
        self.set_class_weights(y_train)

        Xp_train = self.preprocess_input(X_train)
        yp_train = self.preprocess_output(y_train)
        Xp_val = self.preprocess_input(X_val)
        yp_val = self.preprocess_output(y_val)

        time_cb = TimeHistory()
        histories = []

        for run in range(runs):
            self.build_model(learning_rate=0.01)
            history = self.model.fit(
                Xp_train, yp_train,
                validation_data=(Xp_val, yp_val),
                batch_size=BATCH_SIZE,
                epochs=epochs,
                callbacks=self.callbacks + [time_cb],
                class_weight=self.class_weights,
                verbose=2
            )
            ht = history.history
            ht['times'] = time_cb.times
            histories.append(ht)

            print("RUN #", run)
            print('val acc', np.max(ht['val_accuracy']))

        self.history = histories

    def finetune(self, X_train, y_train, X_val, y_val, epochs=200, runs=3):
        if self.weights is None:
            raise ValueError('finetune not available for weights=None')

        self.set_class_weights(y_train)

        Xp_train = self.preprocess_input(X_train)
        Xp_val = self.preprocess_input(X_val)
        yp_train = self.preprocess_output(y_train)
        yp_val = self.preprocess_output(y_val)

        time_cb = TimeHistory()
        histories = []

        for run in range(runs):
            self.build_model(learning_rate=0.001, freeze_backbone=True)
            history0 = self.model.fit(
                Xp_train, yp_train,
                validation_data=(Xp_val, yp_val),
                batch_size=BATCH_SIZE,
                epochs=10,
                callbacks=self.callbacks,
                class_weight=self.class_weights,
                verbose=2
            )

            for l in self.model.layers:
                l.trainable = True
            self.model.compile(
                loss=self.loss,
                optimizer=SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
                metrics=self.metrics)

            history = self.model.fit(
                Xp_train, yp_train,
                validation_data=(Xp_val, yp_val),
                batch_size=BATCH_SIZE,
                epochs=epochs,
                callbacks=self.callbacks + [time_cb],
                class_weight=self.class_weights,
                verbose=2
            )

            ht = history.history
            ht['times'] = time_cb.times
            histories.append(ht)

            print('RUN #', run)
            print('val acc', np.max(ht['val_accuracy']))

        self.history = histories

    def train_top(self, X_train, y_train, X_val, y_val, epochs=200, runs=3):
        self.set_class_weights(y_train)

        Xp_train = self.extract_features(X_train)
        yp_train = self.preprocess_output(y_train)
        Xp_val = self.extract_features(X_val)
        yp_val = self.preprocess_output(y_val)

        inpt_dim = Xp_train.shape[1]
        time_cb = TimeHistory()
        histories = []

        for run in range(runs):
            self.build_top_clf(inpt_dim)
            history = self.clf.fit(
                Xp_train, yp_train,
                validation_data=(Xp_val, yp_val),
                batch_size=BATCH_SIZE,
                epochs=epochs,
                callbacks=self.callbacks + [time_cb],
                class_weight=self.class_weights,
                verbose=2
            )
            ht = history.history
            ht['times'] = time_cb.times
            histories.append(ht)

            print("RUN #", run)
            print('val acc', np.max(ht['val_accuracy']))

        self.history = histories

    def train_lowdata(self, X_train, y_train, X_val, y_val, epochs=200, runs=10):
        # TODO
        raise NotImplementedError()

    def train_catalog(self, X_train, y_train, X_val, y_val, runs=3):
        if len(X_train.shape) != 2 or len(X_val.shape) != 2:
            raise ValueError('X must have shape=2')

        self.set_class_weights(y_train)

        Xp_train = self.preprocess_output(X_train)
        yp_train = self.preprocess_output(y_train)
        Xp_val = self.preprocess_output(X_val)
        yp_val = self.preprocess_output(y_val)

        inpt_dim = Xp_train.shape[1]
        time_cb = TimeHistory()
        histories = []

        for run in range(runs):
            print("RUN #", run)
            self.build_top_clf(inpt_dim)
            history = self.clf.fit(
                Xp_train, yp_train,
                validation_data=(Xp_val, yp_val),
                batch_size=BATCH_SIZE,
                epochs=epochs,
                callbacks=self.callbacks + [time_cb],
                class_weight=self.class_weights,
                verbose=2
            )
            ht = history.history
            ht['times'] = time_cb.times
            histories.append(ht)

        self.history = histories

    def print_history(self):
        print(self.history)

    def dump_history(self):
        if type(self.history) == list:
            hist_tmp = []
            for h in self.history:
                hist_tmp.append(serialize(h))
            self.history = hist_tmp
        else:
            self.history = serialize(self.history)

        if not os.path.exists(os.path.join(self.base_dir, 'history')):
            os.makedirs(os.path.join(self.base_dir, 'history'))
        with open(os.path.join(self.base_dir, 'history', self.model_name+'.json'), 'w') as f:
            json.dump(self.history, f)
        print('dumped history to', os.path.join(self.base_dir, 'history', self.model_name+'.json'))

    def evaluate(self, X, y):
        # TODO FIX
        yp = self.preprocess_output(y)
        if self.backbone is not None:
            Xp = self.preprocess_input(X)
            y_hat = self.model.predict(Xp)
        else:
            Xp = self.preprocess_output(X)
            y_hat = self.clf.predict(Xp)
        compute_metrics(y_hat, yp, target=self.output_type)

    def predict(self, X):
        if self.backbone is not None:
            Xp = self.preprocess_input(X)
            return self.model.predict(Xp)
        else:
            Xp = self.preprocess_output(X)
            return self.clf.predict(Xp)

    def predict_with_embeddings(self, X):
        Xp = self.preprocess_input(X)
        return self.embedder_yx.predict(Xp)

    def extract_features(self, X):
        Xp = self.preprocess_input(X)
        return self.embedder_x.predict(Xp)
