from keras.layers import Input, Conv2D, ReLU, MaxPool2D
from keras.engine.topology import Layer
from keras.initializers import Constant
import keras.backend as K


def vgg16_back300(inpt):

    # conv1
    x = conv_block(inpt, 64, 2)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

    # conv2
    x = conv_block(x, 128, 2)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

    # conv3
    x = conv_block(x, 256, 3)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

    # conv4
    conv4 = conv_block(x, 512, 3)     # x8
    conv4_ = l2Norm(512,10)(conv4)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(conv4)

    # conv5: 3x3 s2 pooling
    x = conv_block(x, 512, 3)
    x = MaxPool2D(pool_size=3, strides=1, padding='same')(x)
    # conv6: atrous conv
    x = Conv2D(1024, 3, strides=1, padding='same', dilation_rate=(6, 6), activation='relu')(x)
    # conv7: 1x1 conv
    conv7 = Conv2D(1024, 1, strides=1, padding='same', activation='relu')(x)
    conv7_ = l2Norm(1024,8)(conv7)

    # conv8: 1x1x256 conv + 3x3x512 s2 conv
    x = Conv2D(256, 1, strides=1, padding='same', activation='relu')(conv7)
    conv8 = Conv2D(512, 3, strides=2, padding='same', activation='relu')(x)

    # conv9: 1x1x128 conv + 3x3x256 s2 conv
    x = Conv2D(128, 1, strides=1, padding='same', activation='relu')(conv8)
    conv9 = Conv2D(256, 3, strides=2, padding='same', activation='relu')(x)

    return [conv4_, conv7_, conv8, conv9]


def vgg16_back512(inpt):

    # conv1
    x = conv_block(inpt, 64, 2)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

    # conv2
    x = conv_block(x, 128, 2)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

    # conv3
    x = conv_block(x, 256, 3)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

    # conv4
    conv4 = conv_block(x, 512, 3)
    conv4_ = l2Norm(512,10)(conv4)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(conv4)

    # conv5: 3x3 s2 pooling
    x = conv_block(x, 512, 3)
    x = MaxPool2D(pool_size=3, strides=1, padding='same')(x)
    # conv6: atrous conv
    x = Conv2D(1024, 3, strides=1, padding='same', dilation_rate=(6, 6), activation='relu')(x)
    # conv7: 1x1 conv
    conv7 = Conv2D(1024, 1, strides=1, padding='same', activation='relu')(x)
    conv7_ = l2Norm(1024,8)(conv7)

    # conv8: 1x1x256 conv + 3x3x512 s2 conv
    x = Conv2D(256, 1, strides=1, padding='same', activation='relu')(conv7)
    conv8 = Conv2D(512, 3, strides=2, padding='same', activation='relu')(x)

    # conv9: 1x1x128 conv + 3x3x256 s2 conv
    x = Conv2D(128, 1, strides=1, padding='same', activation='relu')(conv8)
    conv9 = Conv2D(256, 3, strides=2, padding='same', activation='relu')(x)

    # conv10: 1x1x128 conv + 3x3x256 s2 conv
    x = Conv2D(128, 1, strides=1, padding='same', activation='relu')(conv9)
    conv10 = Conv2D(256, 3, strides=2, padding='same', activation='relu')(x)

    return [conv4_, conv7_, conv8, conv9, conv10]


class l2Norm(Layer):

    def __init__(self, n_channels, norm_scale, **kwargs):
            self.n_channels = n_channels
            self.init_scale = norm_scale
            super(l2Norm, self).__init__(**kwargs)

    def build(self, input_shape):
        self.scale_factor = self.add_weight(name='scale_factor',
                                            shape=(1,1,input_shape[-1]),
                                            initializer=Constant(self.init_scale),
                                            trainable=True)
        super(l2Norm, self).build(input_shape)

    def call(self, x):
        # norm the input
        norm = K.sqrt(K.sum(K.pow(x,2), axis=[1,2], keepdims=True))
        x = x / norm * self.scale_factor
        return x

    def compute_output_shape(self, input_shape):
        return input_shape


def conv_block(x, filters, n_layers, kernel_size=3, strides=1):
    for i in range(n_layers):
        x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
        x = ReLU()(x)
    return x


if __name__ == '__main__':

    inpt = Input((300,300,3))
    features = vgg16_back300(inpt)
    print(features)

    inpt = Input((512,512,3))
    features = vgg16_back512(inpt)
    print(features)








