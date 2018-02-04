from keras.layers import Flatten
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D
from keras import backend as K
from typing import Callable

from k.layers.dense.Dense import MultiDense

from k.blocks.blocks import block_bn_relu, block_conv_bn_relu
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


if __name__ == '__main__':
    from keras.models import Model
    from keras.models import Input
    from keras.layers import Lambda

    from k.layers.residual.Residual import basic

    from keras.datasets import mnist
    from keras.utils import np_utils

    from keras.callbacks import EarlyStopping
    from keras.callbacks import ModelCheckpoint

    # 输入数据
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape([-1, 1, 28, 28])
    y_train = np_utils.to_categorical(y_train)

    x_test = x_test.reshape([-1, 1, 28, 28])
    y_test = np_utils.to_categorical(y_test)

    # 构建模型
    x = Input(name='x', shape=(1, 28, 28))
    net = ResNet(name='resnet',
                 repetitions=[3, 4],
                 block_fn=basic,
                 dense=[
                     {'units': 10, 'activation': 'softmax', 'dropout': None, 'name': 'd'}
                 ]
                 )(x)
    y = Lambda(name='y', function=lambda i: i)(net)
    model = Model(inputs=[x], outputs=[y])
    model.compile(loss='categorical_crossentropy', optimizer='adam',  metrics=['accuracy', 'mse'])

    # 模型训练和保存
    early_stopping = EarlyStopping(monitor='val_acc', min_delta=0.0001, mode='max', patience=3, verbose=1)

    model_checkpoint_better_path = 'mnist.checkpoint.epoch-{epoch:02d}.val_loss-{val_loss:.6f}.val_acc-{val_acc:.6f}'
    model_checkpoint_best_path = 'mnist.checkpoint.best'

    checkpoint_better = ModelCheckpoint(
        model_checkpoint_better_path, save_best_only=True, monitor='val_acc',  mode='max', verbose=1)

    checkpoint_best = ModelCheckpoint(
        model_checkpoint_best_path, save_best_only=True, monitor='val_acc',  mode='max', verbose=1)

    model.fit(x={'x': x_train}, y={'y': y_train}, batch_size=32, epochs=10,
              verbose=1, callbacks=[checkpoint_better, checkpoint_best],
              validation_data=[{'x': x_test}, {'y': y_test}])



