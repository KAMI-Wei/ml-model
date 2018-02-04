from keras.models import Model
from keras.layers import Input,Flatten
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D
from keras import backend as K
from typing import Callable

from k.layers.dense.Dense import MultiDense

from k.blocks.blocks import block_bn_relu, block_conv_bn_relu
from k.layers.residual.Residual import basic
from k.layers.residual.Residual import MultiResidual

from k import ROW_AXIS, COL_AXIS


class ResNet(object):

    def __init__(self,
                 name: str = 'ResNet',
                 block_fn: Callable=None,
                 repetitions: list = None,
                 dense: list = None
                 ):

        self.name = name

        self.repetitions = repetitions

        self.block_fn = block_fn

        self._layer_dense = MultiDense(name=self.name, dense_kwargs_list=dense)

    def __call__(self, inputs, **kwargs):

        conv1 = block_conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(inputs)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", name=self.name+".pool-0")(conv1)

        block = pool1
        filters = 64
        for i, r in enumerate(self.repetitions):
            block = MultiResidual(self.block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
            filters *= 2

        # Last activation
        block = block_bn_relu()(block)

        # Classifier block
        block_shape = K.int_shape(block)
        pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                 strides=(1, 1))(block)
        flatten1 = Flatten(name=self.name+".flatten")(pool2)

        outputs = self._layer_dense(flatten1)

        return outputs

    @staticmethod
    def _residual_block(block_function, filters, repetitions, is_first_layer=False):
        """Builds a residual block with repeating bottleneck blocks.
        """
        def f(inputs):
            for i in range(repetitions):
                init_strides = (1, 1)
                if i == 0 and not is_first_layer:
                    init_strides = (2, 2)
                inputs = block_function(filters=filters,
                                        init_strides=init_strides,
                                        is_first_block_of_first_layer=(is_first_layer and i == 0)
                                        )(inputs)
            return inputs

        return f


if __name__ == '__main__':

    if K.image_dim_ordering() == 'tf':
        x = Input(name='x', shape=(100, 100, 3))
    else:
        x = Input(name='x', shape=(3, 100, 100))

    y = ResNet(name='resnet',
               repetitions=[3, 4, 23, 3],
               block_fn=basic,
               dense=[
                   {'units': 10, 'activation': 'softmax', 'dropout': None, 'kernel_initializer': 'he_normal', 'name': 'd'}
               ]
               )(x)

    model = Model(inputs=[x], outputs=[y])
    model.summary()


