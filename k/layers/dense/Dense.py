from keras.layers import Dense
from keras.layers import Dropout


class MultiDense(object):
    """多层全连接连层


    """

    def __init__(self, name: str = 'dense', dense_kwargs_list: list = None):

        assert name is not None and name is not '', "'name' cannot be empty"
        self.name = name

        if dense_kwargs_list is None:
            dense_kwargs_list = []
        else:
            for layer in dense_kwargs_list:
                assert isinstance(layer, dict)
                assert 'units' in layer and layer['units'] is not None and type(layer['units']) is int
                assert 'activation' in layer and layer['activation'] is not None and type(layer['activation']) is str
                if 'dropout' in layer:
                    assert layer['dropout'] is None or (type(layer['dropout']) is float and (0 < layer['dropout'] < 1))
        self.dense = dense_kwargs_list

    def __call__(self, inputs, **kwargs):

        x_dense = inputs
        for i, kwargs in enumerate(self.dense):
            kwargs_cp = dict(kwargs)
            dropout = None
            if 'dropout' in kwargs_cp:
                dropout = kwargs_cp.pop('dropout')

            if 'name' in kwargs_cp:
                name = kwargs_cp.pop('name')
            else:
                name = 'dense'

            name = self.name + '.' + name + '.' + str(i)
            x_dense = Dense(name=name, **kwargs_cp)(x_dense)
            if dropout is not None:
                x_dense = Dropout(dropout, name=name + '.dropout')(x_dense)

        return x_dense

