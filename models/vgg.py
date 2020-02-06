'''
VGG-like architecture
https://github.com/anokland/local-loss
'''


from keras.layers import Activation, BatchNormalization, Input, Dense, Dropout, Conv2D, Flatten, MaxPooling2D,  ZeroPadding2D
from keras.models import Model


def conv_block(inpt, n_filters, kernel_size=3):
    x =  ZeroPadding2D(1)(inpt)
    x = Conv2D(n_filters, kernel_size)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    return x


def VGG11b(
    input_shape, num_classes, include_top=True, include_features=False, last_activation='softmax', m=2):
    '''
    m (int)     multiplier of number of filters in every conv layer

    conv128-conv128-conv128-conv256-pool-
    conv256-conv512-pool-
    conv512-conv512-pool-
    conv512-pool-fc1024-fc
    '''

    inpt = Input(shape=input_shape)
    x = conv_block(inpt, m*128)
    x = conv_block(x, m*128)
    x = conv_block(x, m*128)
    x = conv_block(x, m*256)
    x = MaxPooling2D(pool_size=2, strides=2)(x)

    x = conv_block(x, m*256)
    x = conv_block(x, m*512)
    x = MaxPooling2D(pool_size=2, strides=2)(x)

    x = conv_block(x, m*512)
    x = conv_block(x, m*512)
    x = MaxPooling2D(pool_size=2, strides=2)(x)

    x = conv_block(x, m*512)
    x = MaxPooling2D(pool_size=2, strides=2)(x)

    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)

    if include_top:
        top = Dense(num_classes, activation=last_activation)(x)
        if include_features:
            return Model(inpt, [top, x])
        else:
            return Model(inpt, top)

    return Model(inpt, x)


if __name__ == '__main__':
    model = VGG11b((32, 32, 3), num_classes=10, include_top=True, last_activation='softmax')
    model.summary()