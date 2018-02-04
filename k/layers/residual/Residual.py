from keras.layers.convolutional import Conv2D
from keras.layers.merge import add
from keras.regularizers import l2
from keras import backend as K

from k import ROW_AXIS, COL_AXIS, CHANNEL_AXIS
from k.blocks.blocks import block_bn_relu_conv

import six


def _get_block(identifier):
    """
    如果输入的是函数名称, 那么根据函数名称从全局区获取实例
    :param identifier:
    :return:
    """
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


def _shortcut(inputs, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(inputs)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = inputs
    # 如果输入和残差块输出的shape不一致, 那么进行1x1的卷积
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(inputs)

    return add([shortcut, residual])


def basic(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """

    def f(inputs):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(inputs)
        else:
            conv1 = block_bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                       strides=init_strides)(inputs)

        residual = block_bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        return _shortcut(inputs, residual)

    return f


def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf

    Returns:
        A final conv layer of filters * 4
    """

    def f(inputs):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                              strides=init_strides,
                              padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(1e-4))(input)
        else:
            conv_1_1 = block_bn_relu_conv(filters=filters, kernel_size=(1, 1),
                                          strides=init_strides)(input)

        conv_3_3 = block_bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)
        residual = block_bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv_3_3)
        return _shortcut(inputs, residual)

    return f


class MultiResidual(object):

    def __init__(self, block_function, filters, repetitions, is_first_layer=False):
        """Builds a residual block with repeating bottleneck blocks.
        """
        self._block_function = _get_block(block_function)
        self._filters = filters
        self._repetitions = repetitions
        self._is_first_layer = is_first_layer

    def __call__(self, inputs):
        for i in range(self._repetitions):
            init_strides = (1, 1)
            if i == 0 and not self._is_first_layer:
                init_strides = (2, 2)
            inputs = self._block_function(filters=self._filters, init_strides=init_strides,
                                          is_first_block_of_first_layer=(self._is_first_layer and i == 0))(inputs)
        return inputs

