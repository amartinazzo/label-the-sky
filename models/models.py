from tensorflow.keras.layers import Conv1D, Conv2D, Dense, Dropout, Flatten, Input
from tensorflow.keras.layers.merge import, add
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
import numpy as np


def dense_net(input_shape, width=64, n_layers=2, n_classes=3):
    inputs = Input(shape=input_shape)
    x = Dense(width, activation='relu')(inputs)
    for i in range(n_layers-1):
        x = Dense(width, activation='relu')(x)
    outputs = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.summary()
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def dimension_reducer(input_dim=512, output_dim=64, n_classes=12, activation='softmax'):
    inputs = Input(shape=input_dim)
    x = Dense(output_dim, activation='relu')(inputs)
    outputs = Dense(n_classes, activation=activation)(x)
    model = Model(inputs, [x, outputs])

    return model


def top_layer_net(input_shape=(512,), nb_classes=3):
    inputs = Input(shape=input_shape)
    outputs = Dense(
        nb_classes, use_bias=False, kernel_regularizer=l2(5e-4),
        kernel_initializer='he_normal', activation='softmax')(inputs)
    model = Model(inputs, outputs)
    return model


def _1d_conv_net(n_filters, kernel_size, strides, input_shape, n_classes):
    model = Sequential()
    model.add(Conv1D(
        filters=n_filters, kernel_size=kernel_size, strides=strides, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Flatten())
    model.add(Dense(n_classes, activation='softmax'))
    model.summary()

    return model
