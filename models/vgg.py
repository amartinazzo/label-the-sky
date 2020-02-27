'''
VGG-like architecture
https://github.com/anokland/local-loss
'''


from keras.layers import Activation, BatchNormalization, Input, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, ZeroPadding2D
from keras.models import Model


def conv_block(inpt, n_filters, kernel_size=3):
    x = ZeroPadding2D(1)(inpt)
    x = Conv2D(n_filters, kernel_size)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    return x


def VGG11b(
    input_shape, num_classes, include_top=True, include_features=False, last_activation='softmax', m=1):
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

    x = conv_block(x, m*1024)
    x = MaxPooling2D(pool_size=4, strides=4)(x)

    x = Flatten()(x)
    # x = Dense(1024, activation='relu')(x)

    if include_top:
        top = Dropout(0.2)(x)
        top = Dense(num_classes, activation=last_activation)(top)
        if include_features:
            return Model(inpt, [top, x])
        else:
            return Model(inpt, top)

    return Model(inpt, x)


def VGG16(input_shape, num_classes, include_top=True, include_features=False, last_activation='softmax'):
    '''
    adapted from source code
    '''
    inpt = Input(shape=input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inpt)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    x = Flatten()(x)

    if include_top:
        top = Dropout(0.2)(x) # do include dropout always!
        top = Dense(num_classes, activation=last_activation)(top)
        if include_features:
            return Model(inpt, [top, x])
        else:
            return Model(inpt, top)
    
    return Model(inpt, x)


if __name__ == '__main__':
    model = VGG16((32, 32, 3), num_classes=12, include_top=True, last_activation='softmax')
    model.summary()
