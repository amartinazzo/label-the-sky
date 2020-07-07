from _datagen import get_dataset, DataGenerator
from efficientnet.tfkeras import EfficientNetB0
from keras_applications import vgg16, resnext
from models.callbacks import TimeHistory
import numpy as np
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD


BACKBONES = ['efficientnet', 'resnext', 'vgg']
CLASS_NAMES = ['GALAXY', 'STAR', 'QSO']
MAG_MAX = 35
N_CHANNELS = [3, 5, 12]
N_CLASSES = 3
OUTPUT_TYPES = ['class', 'magnitudes', 'mockedmagnitudes']
SPLITS = ['train', 'val', 'test']

BACKBONE_FN = {
    'efficientnet': EfficientNetB0,
    'resnext': resnext.ResNeXt50,
    'vgg': vgg16.VGG16
}


def compute_metrics(y_pred, y_true, target='classes', onehot=True):
    if target not in OUTPUT_TYPES:
        raise ValueError('target should be one of %s, but %s was given' % (
            OUTPUT_TYPES, target))

    if target == 'classes':
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


def get_class_weights(df):
    x = 1 / df['class'].value_counts(normalize=True).values
    x = np.round(x / np.max(x), 4)
    return x


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
    def __init__(self, backbone, n_channels, output_type, base_dir, weights):
        if backbone not in BACKBONES:
            raise ValueError('backbone should be one of %s, but %s was given' % (
                BACKBONES, backbone))

        if n_channels not in N_CHANNELS:
            raise ValueError('n_channels should be one of %s, but %s was given' % (
                N_CHANNELS, n_channels))

        if output_type not in OUTPUT_TYPES:
            raise ValueError('output_type should be one of %s, but %s was given' % (
                OUTPUT_TYPES, output_type))

        if weights is not None & weights != 'imagenet' & not os.path.exists(weights):
            raise ValueError('weights must be: None, imagenet, or a valid path to a h5 file')

        self.backbone = backbone
        self.n_channels = n_channels
        self.output_type = output_type
        self.save_dir = save_dir
        self.weights = weights

        self.input_shape = (32, 32, n_channels)

        self.save_dir = os.path.join(base_dir, 'trained_models')

        self.data_dir = 'crops_rgb32' if n_channels==3 else 'crops_calib'
        self.data_dir = os.path.join(base_dir, self.data_dir)

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

        self.timestamp = 0  #TODO
        self.model_name = 0  #TODO

    def build_dataset(self, df, split='train', shuffle=True, bs=32):
        if split not in SPLITS:
            raise ValueError('split should be one of %s, but %s was given' % (
                SPLITS, split))

        df = df[df.split == split]
        ids, y, labels = get_dataset(df, target=self.output_type, n_bands=self.n_channels)

        shuffle = False if split != 'train' else shuffle
        batch_size = 1 if split == 'test' else bs

        params = {
            'batch_size': batch_size,
            'data_folder': self.data_dir,
            'input_dim': self.input_shape,
            'n_outputs': self.n_outputs,
            'target': self.output_type,
            'shuffle': shuffle,
        }

        data_gen = DataGenerator(ids, labels=labels, **params)

        return data_gen, y

    def build_model(self):
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

        x = self.model.output
        x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)

        y = Dense(1024, activation='relu')(x)
        y = Dropout(0.5)(y)
        y = Dense(self.n_outputs, activation=self.activation)(y)

        self.top_layer_idx = -3
        self.embedder_x = Model(inputs=self.model.input, outputs=x)
        self.embedder_yx = Model(inputs=self.model.input, outputs=[y, x])
        self.model = Model(inputs=self.model.input, outputs=y)

        for l in self.model.layers:
            l.trainable = True

        if os.path.exists(self.weights):
            self.model.load_weights(self.weights, by_name=True, skip_mismatch=True)
            print('loaded weights')

        self.set_callbacks()

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

    def pretrain(self, gen_train, gen_val, epochs=500):
        if self.output_type == 'class':
            raise ValueError('pretraining not available for classes')

        opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss=self.loss, optimizer=opt)

        time_cb = TimeHistory()

        history = self.model.fit_generator(
            generator=gen_train,
            validation_data=gen_val,
            epochs=epochs,
            callbacks=self.callbacks + [time_cb],
            verbose=2
        )

        hist = history.history
        hist['times'] = time_cb.times

        self.history = hist

    def finetune(self, gen_train, gen_val):
        if self.weights is None:
            raise ValueError('finetune not available for weights=None')

        opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss=self.loss, optimizer=opt)

        time_cb = TimeHistory()
        histories = []

        weights0 = self.model.get_weights()
        for run in range(runs):
            self.model.set_weights(weights0)

            for layer in self.model.layers[:self.top_layer_idx]:
                layer.trainable = False

            history0 = self.model.fit(
                generator=gen_train,
                validation_data=gen_val,
                epochs=10,
                callbacks=self.callbacks,
                class_weight=class_weights,
                verbose=2
            )

            for layer in self.model.layers:
                layer.trainable = True

            history = clf.fit(
                generator=gen_train,
                validation_data=gen_val,
                epochs=epochs,
                callbacks=self.callbacks + [time_cb],
                class_weight=class_weights,
                verbose=2
            )

            ht = history.history
            ht['times'] = time_cb.times
            histories.append(ht)

            self.history = histories

    def train_clf(self, gen_train, y_train, gen_val, y_val, class_weights=None, epochs=200, runs=3):
        opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss=self.loss, optimizer=opt)

        time_cb = TimeHistory()
        histories = []

        Xf_train = self.extract_features(gen_train)
        Xf_val = self.extract_features(gen_val)

        inpt = Input(shape=Xf_train.shape[:-1])
        x = Dense(12, activation='relu')(inpt)
        x = Dense(N_CLASSES, activation=self.activation)(x)

        self.clf = Model(inpt, x)

        self.clf.compile(loss=self.loss, optimizer=opt, metrics=['accuracy'])

        weights0 = clf.get_weights()
        for run in range(runs):
            self.clf.set_weights(weights0)
            history = self.clf.fit(
                Xf_train,
                y_train,
                validation_data=(Xf_val, y_val),
                epochs=epochs,
                callbacks=self.callbacks + [time_cb],
                class_weight=class_weights,
                verbose=2
            )
            ht = history.history
            ht['times'] = time_cb.times
            histories.append(ht)

        self.history = histories

    def train_clf_lowdata(self, gen_train, gen_val, class_weights=None, epochs=200, runs=10):
        # TODO
        raise NotImplementedError()

    def print_history(self):
        print(self.history)

    def dump_history(self):
        if not os.path.exists(os.path.join(base_dir, 'history')):
            os.makedirs(os.path.join(base_dir, 'history'))
        with open(os.path.join(base_dir, 'history', self.model_name+'.json'), 'w') as f:
            json.dump(serialize(self.history), f)
        print('dumped history to', os.path.join(base_dir, 'history', self.model_name+'.json'))

    def evaluate(self, gen, y):
        y_hat = self.model.predict_generator(gen)
        compute_metrics(y_hat, y, target=self.output_type)

    def predict(self, gen):
        return self.model.predict_generator(gen)

    def predict_with_embeddings(self, gen):
        return self.embedder_yx.predict_generator(gen)

    def extract_features(self, gen):
        return self.embedder_x.predict_generator(gen)
