from efficientnet.tfkeras import EfficientNetB0
from keras_applications import vgg16, resnext
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, Flatten, LeakyReLU, Activation, GlobalAveragePooling2D, Dropout


BACKBONES = ['efficientnet', 'resnext', 'vgg']
N_CHANNELS = [3, 5, 12]
OUTPUT_TYPES = ['class', 'magnitudes', 'mockedmagnitudes']

BACKBONE_FN = {
    'efficientnet': EfficientNetB0,
    'resnext': resnext.ResNeXt50,
    'vgg': vgg16.VGG16
}


def relu_saturated(x):
    return keras.backend.relu(x, max_value=1.)


def set_random_seeds():
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)
    tf.random.set_seed(420)


class Trainer:
    def __init__(self, backbone, n_channels, output_type, weights, freeze):
        if backbone not in BACKBONES:
            raise ValueError('backbone should be one of %s, but %s was given' % (
                BACKBONES, backbone))

        if n_channels not in N_CHANNELS:
            raise ValueError('n_channels should be one of %s, but %s was given' % (
                N_CHANNELS, n_channels))

        if output_type not in OUTPUT_TYPES:
            raise ValueError('output_type should be one of %s, but %s was given' % (
                N_CHANNELS, n_channels))

        if weights is not None & weights != 'imagenet' & not os.path.exists(weights):
            raise ValueError('weights must be: None, imagenet, or a valid path to a h5 file')

        if type(freeze) != bool:
            raise ValueError('freeze must be a boolean')

        self.backbone = backbone
        self.n_channels = n_channels
        self.output_type = output_type
        self.weights = weights

        self.input_shape = (32, 32, n_channels)

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

        self.embedder_x = Model(inputs=self.model.input, outputs=x)
        self.embedder_yx = Model(inputs=self.model.input, outputs=[y, x])
        self.model = Model(inputs=self.model.input, outputs=y)

        if self.weights is not None and 'magnitudes' in self.weights:
            self.model.load_weights(self.weights, by_name=True, skip_mismatch=True)
            print('loaded weights')

        optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss=self.loss, optimizer=optimizer, metrics=self.metrics)

    def print_summary(self):
        self.model.summary()

    def pretrain(self, gen_train, gen_val, epochs=500, verbose=True):
        time_callback = TimeHistory()
        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1),
            ModelCheckpoint(model_file, monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min'),
            EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1),
            time_callback
        ]

        history = self.model.fit_generator(
            generator=gen_train,
            validation_data=gen_val,
            # steps_per_epoch=len(gen_train),
            # validation_steps=len(gen_val),
            epochs=epochs,
            callbacks=callbacks,
            verbose=2
        )

        if verbose:
            print('History')
            print(history.history)
            print('Time taken per epoch (s)')
            print(time_callback.times)

        return history

    def finetune(self):
        pass

    def train_top_clf(self):
        pass

    def run_inner_clf_train(self, gen_train, gen_val, class_weights=None, runs=3):
        accuracies = []
        weights0 = self.model.get_weights()
        for run in range(runs):
            self.model.set_weights(weights0)
            history = self.model.fit_generator(
                generator=gen_train,
                validation_data=gen_val,
                # steps_per_epoch=len(gen_train),
                # validation_steps=len(gen_val),
                epochs=epochs,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=2
            )
            acc = np.max(np.array(history.history['val_acc']))
            accuracies.append(acc)

    def train_clf(self, gen_train, gen_val):
        if self.freeze:
            Xf_train = self.extract_features(gen_train)
            Xf_val = self.extract_features(gen_val)
            # TODO
            # add fit top layer with or without imagenet
        else:
            self.run_inner_clf_train(gen_train, gen_val, class_weights)

    def predict(self, X):
        return self.model.predict(X)

    def predict_with_embeddings(self, X):
        return self.embedder_yx.predict(X)

    def extract_features(self, X):
        return self.embedder_x.predict(X)
