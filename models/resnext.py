'''
ResNeXt
https://github.com/verbpronoun/keras-dl-benchmark/blob/master/resnext_builder.py
'''


from keras.layers.core import Activation, Dense, Dropout, Lambda, Flatten
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.pooling import GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D, MaxPooling1D
from keras.layers import Input, Dense, Activation, Flatten
from keras.layers import Convolution2D, AveragePooling2D, BatchNormalization, MaxPooling2D, ZeroPadding2D
from keras.layers.merge import concatenate, add
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.utils import np_utils
import keras.backend as K
import numpy as np


def my_conv(input, num_filters, kernel_size_tuple, strides=1, padding='valid'):
    x = Convolution2D(num_filters, kernel_size_tuple, strides=strides, padding=padding, 
                      use_bias=True, kernel_initializer='he_normal')(input)
    return x


def Block(input, numFilters, stride, isConvBlock, cardinality, bottleneck_width):
    expansion = 4

    width = int((numFilters * bottleneck_width) / 64)
    group = []

    for i in range(cardinality):
        # make grouped convolution
        x = my_conv(input, width, (1, 1), strides=stride)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)

        x = my_conv(x, width, (3, 3), padding='same')
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)

        group.append(x)

    x = concatenate(group, axis=channel_axis)
    x = my_conv(x, expansion * numFilters, (1, 1))
    x = BatchNormalization(axis=channel_axis)(x)

    if isConvBlock:
        shortcut = my_conv(input, expansion * numFilters, (1, 1), strides=stride)
        shortcut = BatchNormalization(axis=channel_axis)(shortcut)
    else:
        shortcut = input

    x = add([x, shortcut])
    x = Activation('relu')(x)

    return x


def make_layer(block, input, numFilters, numBlocks, stride, cardinality, bottleneck_width):
    x = block(input, numFilters, stride, True, cardinality, bottleneck_width)
    for i in range(numBlocks - 1):
        x = block(x, numFilters, 1, False, cardinality, bottleneck_width)
    return x


def ResNeXt_builder(
    block, num_blocks, input_shape, num_classes, cardinality, bottleneck_width, last_activation, output_dim):
    img_input = Input(shape=input_shape)
    x = my_conv(img_input, 64, (1, 1))
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    
    x = make_layer(Block, x, 64, num_blocks[0], 1, cardinality, bottleneck_width)
    x = make_layer(Block, x, 128, num_blocks[1], 2, cardinality, bottleneck_width)
    x = make_layer(Block, x, 256, num_blocks[2], 2, cardinality, bottleneck_width)
    # x = make_layer(Block, x, 512, num_blocks[3], 2, cardinality, bottleneck_width)
    
    x = AveragePooling2D((8, 8), strides=8)(x)
    x = Flatten()(x)

    if output_dim is not None:
        x = Dense(output_dim, activation='relu')(x)

    top = Dense(num_classes, activation=last_activation)(x)
    
    return Model(img_input, [top, x])


def ResNeXt29(input_shape, num_classes, width=64, cardinality=8, last_activation='softmax', output_dim=None):
    depth_seq = (3,3,3)
    return ResNeXt_builder(
        Block, depth_seq, input_shape, num_classes, cardinality, width, last_activation, output_dim)


if __name__ == '__main__':
    model = ResNeXt29((32, 32, 3), num_classes=3, width=64, cardinality=4)
    model.summary()