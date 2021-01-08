# resnet 50 & 101
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, ReLU, add


n_blocks = {50: [3,4,6,3], 101: [3,4,23,3]}
n_filters = [256, 512, 1024, 2048]


def resnet_back(inpt, depth=50):

    # stem: conv+bn+relu+pool
    x = Conv_BN(inpt, 64, 7, strides=2, activation='leaky')
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # conv block1
    x = res_block(x, n_filters[0], strides=1)
    # conv block2
    conv2 = res_block(x, n_filters[1], strides=2)       # x8
    # conv block3
    conv3 = res_block(conv2, n_filters[2], strides=2)
    # conv block4
    x = res_block(conv3, n_filters[3], strides=2)
    conv4 = Conv2D(512, 3, strides=1, padding='same')(x)

    # conv6
    x = Conv2D(256, 3, strides=1, padding='same', activation='relu')(x)
    conv5 = Conv2D(256, 1, strides=2, padding='same', activation='relu')(x)

    return [conv2, conv3, conv4, conv5]


def res_block(x, n_filters, strides):
    inpt = x
    # residual
    x = Conv_BN(x, n_filters//4, 1, strides=strides, activation='relu')
    x = Conv_BN(x, n_filters//4, 3, strides=1, activation='relu')
    x = Conv_BN(x, n_filters, 1, strides=1, activation=None)
    # shortcut
    if strides!=1 or inpt._keras_shape[-1]!=n_filters:
        inpt = Conv_BN(inpt, n_filters, 1, strides=strides, activation=None)
    x = add([inpt, x])
    x = ReLU()(x)
    return x


def Conv_BN(x, n_filters, kernel_size, strides, activation=None):
    x = Conv2D(n_filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    if activation:
        x = ReLU()(x)
    return x


if __name__ == '__main__':

    inpt = Input(input_shape=(224,224,3))
    model = resnet_back(inpt, depth=50)
    model.summary()
    # model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")

