import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers
import numpy as np
#tf.enable_eager_execution() #added. so as to be able to use numpy arrays easily

def limit_mem():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.InteractiveSession(config=config)


class PeriodicPadding2D(tf.keras.layers.Layer):
    def __init__(self, pad_width, **kwargs):
        super().__init__(**kwargs)
        self.pad_width = pad_width

    def call(self, inputs, **kwargs):
        if self.pad_width == 0:
            return inputs
        inputs_padded = tf.concat(
            [inputs[:, :, -self.pad_width:, :], inputs, inputs[:, :, :self.pad_width, :]], axis=2)
        # Zero padding in the lat direction
        inputs_padded = tf.pad(inputs_padded, [[0, 0], [self.pad_width, self.pad_width], [0, 0], [0, 0]])
        return inputs_padded

    def get_config(self):
        config = super().get_config()
        config.update({'pad_width': self.pad_width})
        return config


class PeriodicConv2D(tf.keras.layers.Layer):
    def __init__(self, filters,
                 kernel_size,
                 conv_kwargs={},
                 **kwargs, ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv_kwargs = conv_kwargs
        if type(kernel_size) is not int:
            assert kernel_size[0] == kernel_size[1], 'PeriodicConv2D only works for square kernels'
            kernel_size = kernel_size[0]
        pad_width = (kernel_size - 1) // 2
        self.padding = PeriodicPadding2D(pad_width)
        self.conv = Conv2D(
            filters, kernel_size, padding='valid', **conv_kwargs
        )

    def call(self, inputs):
        return self.conv(self.padding(inputs))

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters, 'kernel_size': self.kernel_size, 'conv_kwargs': self.conv_kwargs})
        return config


class ChannelSlice(tf.keras.layers.Layer):

    def __init__(self, n_out, **kwargs):
        self.n_out = n_out
        super().__init__(**kwargs)

    def _slice(self, inputs):
        # Input: [samples, lat, lon, filters]
        return inputs[..., :self.n_out]

    def __call__(self, inputs):
        out = Lambda(self._slice)(inputs)
        return out
    
def convblock(inputs, filters, kernel=3, stride=1, bn_position=None, l2=0,
              use_bias=True, dropout=0, activation='relu'):
    x = inputs
    if bn_position == 'pre': x = BatchNormalization()(x)
    x = PeriodicConv2D(
        filters, kernel, conv_kwargs={
            'kernel_regularizer': regularizers.l2(l2),
            'use_bias': use_bias
        }
    )(x)
    if bn_position == 'mid': x = BatchNormalization()(x)
    x = LeakyReLU()(x) if activation == 'leakyrelu' else Activation(activation)(x) 
    if bn_position == 'post': x = BatchNormalization()(x)
    if dropout > 0: x = Dropout(dropout)(x)
    return x

def resblock(inputs, filters, kernel, bn_position=None, l2=0, use_bias=True,
             dropout=0, skip=True, activation='relu'):
    x = inputs
    for _ in range(2):
        x = convblock(
            x, filters, kernel, bn_position=bn_position, l2=l2, use_bias=use_bias,
            dropout=dropout, activation=activation
        )
    if skip: x = Add()([inputs, x])
    return x



def build_resnet(filters, kernels, input_shape, bn_position=None, use_bias=True, l2=0,
                 skip=True, dropout=0, activation='relu'):
    x = input = Input(shape=input_shape)

    # First conv block to get up to shape
    x = convblock(
        x, filters[0], kernels[0], bn_position=bn_position, l2=l2, use_bias=use_bias,
        dropout=dropout, activation=activation
    )

    # Resblocks
    for f, k in zip(filters[1:-1], kernels[1:-1]):
        x = resblock(x, f, k, bn_position=bn_position, l2=l2, use_bias=use_bias,
                dropout=dropout, skip=skip, activation=activation)

    # Final convolution
    output = PeriodicConv2D(
        filters[-1], kernels[-1], conv_kwargs={'kernel_regularizer': regularizers.l2(l2)})(x)
    return keras.models.Model(input, output)


def build_unet(input_shape, n_layers, filters_start, channels_out, kernel=3, u_skip=True,
               res_skip=True, l2=0, bn_position=None, dropout=0):
    "https://github.com/Nishanksingla/UNet-with-ResBlock/blob/master/resnet34_unet_model.py"
    x = input = Input(shape=input_shape)
    filters = filters_start

    # Down
    down_layers = []
    for i in range(n_layers):
        # Resblock
        x_res = PeriodicConv2D(
            filters, 1, conv_kwargs={
                'use_bias': False, 'kernel_regularizer': regularizers.l2(l2)})(x)
        x = convblock(x, filters, kernel, bn_position=bn_position, l2=l2, dropout=dropout)
        x = convblock(x, filters, kernel, bn_position=bn_position, l2=l2, activation='linear',
                      dropout=dropout)
        if res_skip: x = Add()([x, x_res])
        x = ReLU()(x)
        if not i == n_layers - 1:
            down_layers.append(x)
            # Downsampling
            x = MaxPooling2D()(x)
            filters *= 2

    # Up
    for dl in reversed(down_layers):
        filters //= 2
        # Upsample
        x = UpSampling2D()(x)
        x = PeriodicConv2D(filters, 3, conv_kwargs={'kernel_regularizer': regularizers.l2(l2)})(x)
        x = ReLU()(x)

        # Concatenate
        if u_skip:
            x = Concatenate()([x, dl])

        # Resblock
        x_res = PeriodicConv2D(filters, 1, conv_kwargs={'use_bias': False})(x)
        x = convblock(x, filters, kernel, bn_position=bn_position, l2=l2, dropout=dropout)
        x = convblock(x, filters, kernel, bn_position=bn_position, l2=l2, activation='linear',
                      dropout=dropout)
        if res_skip: x = Add()([x, x_res])
        x = ReLU()(x)

    x = PeriodicConv2D(channels_out, 1, conv_kwargs={'kernel_regularizer': regularizers.l2(l2)})(x)
    return keras.models.Model(input, x)


def create_lat_mse(lat):
    weights_lat = np.cos(np.deg2rad(lat)).values
    weights_lat /= weights_lat.mean()
    def lat_mse(y_true, y_pred):
        error = y_true - y_pred
        mse = (error)**2 * weights_lat[None, : , None, None]
        return mse
    return lat_mse


# Agrawal et al version
def basic_block(x, filters, dropout):
    shortcut = x
    x = PeriodicConv2D(filters, kernel_size=3)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = PeriodicConv2D(filters, kernel_size=3)(x)
    if dropout > 0: x = Dropout(dropout)(x)

    shortcut = PeriodicConv2D(filters, kernel_size=3)(shortcut)
    return Add()([x, shortcut])


def downsample_block(x, filters, dropout):
    shortcut = x
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D()(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = PeriodicConv2D(filters, kernel_size=3)(x)
    if dropout > 0: x = Dropout(dropout)(x)

    shortcut = PeriodicConv2D(filters, kernel_size=3, conv_kwargs={'strides': 2})(shortcut)
    return Add()([x, shortcut])


def upsample_block(x, from_down, filters, dropout):
    x = Concatenate()([x, from_down])
    x = UpSampling2D()(x)
    shortcut = x

    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = PeriodicConv2D(filters, kernel_size=3)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = PeriodicConv2D(filters, kernel_size=3)(x)
    if dropout > 0: x = Dropout(dropout)(x)

    shortcut = PeriodicConv2D(filters, kernel_size=3)(shortcut)
    return Add()([x, shortcut])


def build_unet_google(filters, input_shape, output_channels, dropout=0):
    inputs = x = Input(input_shape)
    x = basic_block(x, filters[0], dropout=dropout)

    # Encoder
    from_down = []
    for f in filters[:-1]:
        x = downsample_block(x, f, dropout=dropout)
        from_down.append(x)

    # Bottleneck
    x = basic_block(x, filters[-1], dropout=dropout)

    # Decoder
    for f, d in zip(filters[:-1][::-1], from_down[::-1]):
        x = upsample_block(x, d, f, dropout=dropout)

    # Final
    outputs = PeriodicConv2D(output_channels, kernel_size=1)(x)

    return keras.models.Model(inputs, outputs)