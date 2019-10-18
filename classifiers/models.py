from keras.layers.advanced_activations import PReLU
from keras.layers.core import Activation, Dense, Dropout, Lambda, Flatten
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.pooling import GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D, MaxPooling1D
from keras.layers import Input
from keras.layers.merge import concatenate, add
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.regularizers import l2
import keras.backend as K


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
    outputs = Dense(nb_classes, use_bias=False, kernel_regularizer=l2(5e-4),
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


def resnext(input_shape=None, depth=29, cardinality=8, width=64, weight_decay=5e-4,
            top_layer=True, input_tensor=None, pooling='avg', classes=3, last_activation='softmax',
            output_dim=512):
    '''
    resnext adapted from https://github.com/titu1994/Keras-ResNeXt/blob/master/resnext.py

    Instantiate the ResNeXt architecture. Note that ,
    when using TensorFlow for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.
    # Arguments
        depth: number or layers in the ResNeXt model. Can be an
            integer or a list of integers.
        cardinality: the size of the set of transformations
        width: multiplier to the ResNeXt width (number of filters)
        weight_decay: weight decay (l2 norm)
        top_layer: include top layer (boolean)
        weights: `None` (random initialization)
    # Returns
        A Keras model instance.

    '''

    if type(depth) == int:
        if (depth - 2) % 9 != 0:
            raise ValueError('Depth of the network must be such that (depth - 2)'
                             'should be divisible by 9.')

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = __create_res_next(classes, img_input, top_layer, depth, cardinality, width,
                          weight_decay, pooling, last_activation, output_dim)

    model = Model(img_input, x, name='resnext')

    return model


def __initial_conv_block(input_layer, weight_decay=5e-4):
    '''
    Adds an initial convolution block, with batch normalization and relu activation
    Args:
        input: input tensor
        weight_decay: weight decay factor
    Returns: a keras tensor
    '''
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(input_layer)
    x = BatchNormalization(axis=channel_axis)(x)
    x = PReLU()(x) #Activation('relu')(x)

    return x


def __grouped_convolution_block(input_layer, grouped_channels, cardinality, strides, weight_decay=5e-4):
    '''
    Adds a grouped convolution block. It is an equivalent block from the paper
    Args:
        input: input tensor
        grouped_channels: grouped number of filters
        cardinality: cardinality factor describing the number of groups
        strides: performs strided convolution for downscaling if > 1
        weight_decay: weight decay term
    Returns: a keras tensor
    '''
    init = input_layer
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    group_list = []

    if cardinality == 1:
        # with cardinality 1, it is a standard convolution
        x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
        x = BatchNormalization(axis=channel_axis)(x)
        x = PReLU()(x) #Activation('relu')(x)
        return x

    for c in range(cardinality):
        x = Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels]
        if K.image_data_format() == 'channels_last' else
        lambda z: z[:, c * grouped_channels:(c + 1) * grouped_channels, :, :])(input_layer)

        x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)

        group_list.append(x)

    group_merge = concatenate(group_list, axis=channel_axis)
    x = BatchNormalization(axis=channel_axis)(group_merge)
    x = PReLU()(x) #Activation('relu')(x)

    return x


def __bottleneck_block(input_layer, filters=64, cardinality=8, strides=1, weight_decay=5e-4):
    '''
    Adds a bottleneck block
    Args:
        input: input tensor
        filters: number of output filters
        cardinality: cardinality factor described number of
            grouped convolutions
        strides: performs strided convolution for downsampling if > 1
        weight_decay: weight decay factor
    Returns: a keras tensor
    '''
    init = input_layer

    grouped_channels = int(filters / cardinality)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    # Check if input number of filters is same as 16 * k, else create convolution2d for this input
    if K.image_data_format() == 'channels_first':
        if init._keras_shape[1] != 2 * filters:
            init = Conv2D(filters * 2, (1, 1), padding='same', strides=(strides, strides),
                          use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
            init = BatchNormalization(axis=channel_axis)(init)
    else:
        if init._keras_shape[-1] != 2 * filters:
            init = Conv2D(filters * 2, (1, 1), padding='same', strides=(strides, strides),
                          use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
            init = BatchNormalization(axis=channel_axis)(init)

    x = Conv2D(filters, (1, 1), padding='same', use_bias=False,
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(input_layer)
    x = BatchNormalization(axis=channel_axis)(x)
    x = PReLU()(x) #Activation('relu')(x)

    x = __grouped_convolution_block(x, grouped_channels, cardinality, strides, weight_decay)

    x = Conv2D(filters * 2, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=channel_axis)(x)

    x = add([init, x])
    x = PReLU()(x) #Activation('relu')(x)

    return x


def __create_res_next(nb_classes, img_input, top_layer, depth=29, cardinality=8, width=4,
    weight_decay=5e-4, pooling=None, last_activation='softmax', output_dim=512):
    '''
    Creates a ResNeXt model with specified parameters
    Args:
        nb_classes: Number of output classes
        img_input: Input tensor or layer
        depth: Depth of the network. Can be an positive integer or a list
               Compute N = (n - 2) / 9.
               For a depth of 56, n = 56, N = (56 - 2) / 9 = 6
               For a depth of 101, n = 101, N = (101 - 2) / 9 = 11
        cardinality: the size of the set of transformations.
               Increasing cardinality improves classification accuracy,
        width: Width of the network.
        weight_decay: weight_decay (l2 norm)
        pooling: Optional pooling mode for feature extraction there is no top layer.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
    Returns: a Keras Model
    '''

    if type(depth) is list or type(depth) is tuple:
        # If a list is provided, defer to user how many blocks are present
        N = list(depth)
    else:
        # Otherwise, default to 3 blocks each of default number of group convolution blocks
        N = [(depth - 2) // 9 for _ in range(3)]

    filters = cardinality * width
    filters_list = []

    for i in range(len(N)):
        filters_list.append(filters)
        filters *= 2  # double the size of the filters

    x = __initial_conv_block(img_input, weight_decay)

    # block 1 (no pooling)
    for i in range(N[0]):
        x = __bottleneck_block(x, filters_list[0], cardinality, strides=1, weight_decay=weight_decay)

    N = N[1:]  # remove the first block from block definition list
    filters_list = filters_list[1:]  # remove the first filter from the filter list

    # block 2 to N
    for block_idx, n_i in enumerate(N):
        for i in range(n_i):
            if i == 0:
                x = __bottleneck_block(x, filters_list[block_idx], cardinality, strides=2,
                                       weight_decay=weight_decay)
            else:
                x = __bottleneck_block(x, filters_list[block_idx], cardinality, strides=1,
                                       weight_decay=weight_decay)

    if pooling == 'avg':
        x = GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D()(x)

    if output_dim != 512: # TODO fix hard coded number
        x = Dense(output_dim, use_bias=False, kernel_regularizer=l2(weight_decay),
                  kernel_initializer='he_normal')(x)

    if top_layer:
        x = PReLU()(x)
        x = Dense(nb_classes, use_bias=False, kernel_regularizer=l2(weight_decay),
                  kernel_initializer='he_normal', activation=last_activation)(x)

    return x


if __name__ == '__main__':
    model = resnext((32, 32, 3), depth=29, cardinality=8, width=64)
    model.summary()