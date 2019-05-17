"""
an implementation of a convolutional autoencoder using keras
adapted from https://github.com/jmmanley/conv-autoencoder/blob/master/cae.py
"""
from keras.callbacks import BaseLogger, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Input, InputLayer, MaxPooling2D, Reshape
from keras.models import Model, Sequential
import numpy as np
import os

class ConvAutoEncoder:
    # filter dims assume input_shape is mxnxc, and m,n are multiples of 2
    def __init__(self, input_shape, output_dim, filters=[32, 64, 128, 256],
                 kernel=(3,3), stride=(1,1), strideundo=2, pool=(2,2),
                 optimizer="adamax", lossfn="mse"):
        self.input_shape = input_shape
        self.output_dim  = output_dim

        n_filters = len(filters)

        # define encoder architecture
        self.encoder = Sequential()
        self.encoder.add(InputLayer(input_shape))
        for i in range(len(filters)):
            self.encoder.add(Conv2D(filters=filters[i], kernel_size=kernel, strides=stride, activation='elu', padding='same'))
            self.encoder.add(MaxPooling2D(pool_size=pool))
        self.encoder.add(Flatten())
        # self.encoder.add(Dense(output_dim))

        # define decoder architecture
        self.decoder = Sequential()
        self.decoder.add(InputLayer((1024,))) # TODO generalize for any output_dim
        # self.decoder.add(Dense(int(filters[len(filters)-1] * input_shape[0]/(2**(len(filters))) * input_shape[1]/(2**(len(filters))))))
        self.decoder.add(Reshape((int(input_shape[0]/2**n_filters), int(input_shape[1]/2**n_filters), filters[n_filters-1])))
        for i in range(1,len(filters)):
            self.decoder.add(Conv2DTranspose(filters=filters[len(filters)-i], kernel_size=kernel, strides=strideundo, activation='elu', padding='same'))
        self.decoder.add(Conv2DTranspose(filters=input_shape[2], kernel_size=kernel, strides=strideundo, activation=None, padding='same'))

        # compile model
        input         = Input(input_shape)
        encoded       = self.encoder(input)
        reconstructed = self.decoder(encoded)

        self.ae = Model(inputs=input, outputs=reconstructed)
        self.ae.compile(optimizer=optimizer, loss=lossfn)
        self.encoder.summary()
        self.decoder.summary()


    def fit_generator(self, train_gen, val_gen, epochs=30):
        callbacks=[
            ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=10, min_lr=1e-6),
            BaseLogger()
            ]

        self.ae.fit_generator(        
            generator=train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=2)
        # self.mse = self.ae.evaluate(test, test)
        # print('CAE MSE on validation data: ', self.mse)


    def save_weights(self, path=None, prefix=""):
        if path is None: path = os.getcwd()
        self.encoder.save_weights(os.path.join(path, prefix + "encoder_weights.h5"))
        self.decoder.save_weights(os.path.join(path, prefix + "decoder_weights.h5"))


    def load_weights(self, path=None, prefix=""):
        if path is None: path = os.getcwd()
        self.encoder.load_weights(os.path.join(path, prefix + "encoder_weights.h5"))
        self.decoder.load_weights(os.path.join(path, prefix + "decoder_weights.h5"))


    def encode(self, data_generator):
        return self.encoder.predict_generator(data_generator)


    def decode(self, encoded):
        return self.decoder.predict(encoded)


if __name__=='__main__':
    import os,sys,inspect
    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    import datagen

    mode='train'
    
    ae = ConvAutoEncoder(input_shape=(32,32,12), output_dim=1024)

    home_path = os.path.expanduser('~')
    params = {
        'data_folder': home_path+'/raw-data/crops/normalized/', 'dim': (32,32,12),
        'n_classes': 3, 'batch_size': 512, 'mode':'autoencoder'}

    X_train, _, _ = datagen.get_sets(home_path+'/raw-data/matched_cat_dr1_full_train.csv')
    X_val, _, _ = datagen.get_sets(home_path+'/raw-data/matched_cat_dr1_full_val.csv')
    print('train size', len(X_train))
    print('val size', len(X_val))
    train_generator = datagen.DataGenerator(X_train, **params)
    val_generator = datagen.DataGenerator(X_val, **params)

    ### debugging

    print('encoding')
    X_encoded = ae.encode(val_generator)
    print(len(X_encoded))
    print(X_encoded[0].shape)

    print('decoding')
    X_decoded = ae.decode(X_encoded)
    print(len(X_decoded))
    print(X_decoded[0].shape)

    ### end debugging

    if mode=='train':
        # make only 1 gpu visible
        os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES']='0'

        ae.fit_generator(train_generator, val_generator)
        ae.save_weights()
