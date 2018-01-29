from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding


class EmbeddingLSTM(object):
    """
    经典 LSTM 应用模型:
    ================================================================================
      Embedding
    ================================================================================
      LSTM
    ================================================================================
      Dense
    ================================================================================

    通过工厂模式兼容为 keras 的 layer

    TODO: 继承 Layer 类, 实现真正意义上的 keras.layer

    # Arguments
        name: 模型名称, 模型各层的命名规则为 [模型名称].[层名称]
        input_length: 输入序列的长度
        embedding_word_number: Embedding层的词典大小
        embedding_vector_length: Embedding层的输出序列长度
        embedding_dropout: Embedding层的dropout. 若为None或1, 则无该Dropout层.
        embedding_kwargs: Embedding层的其他参数, 参考 keras.layers.embeddings.Embedding
        lstm_units: LSTM层的神经元数
        lstm_dropout: LSTM层的dropout. 若为None或1, 则无该Dropout层.
        lstm_kwargs: LSTM层的其他参数, 参考 keras.layers.LSTM
        dense: 定义每个全连接层的名称, 神经元数, 参数初始化函数, 激活函数, dropout等. 若为None或[], 则无全连接层

    """

    def __init__(self,
                 name: str = 'EmbeddingLSTM',

                 embedding_word_number: int = None,
                 embedding_vector_length: int = None,
                 embedding_dropout: float = None,
                 embedding_kwargs: dict = None,

                 lstm_units: int = None,
                 lstm_dropout: float = None,
                 lstm_kwargs: dict = None,

                 dense: list = None

                 ):
        """
        构造函数, 设置模型参数
        :param name: 模型各层的命名规则为 [模型名称].[层名称]
        :param embedding_word_number: Embedding层的词典大小
        :param embedding_vector_length: Embedding层的输出序列长度
        :param embedding_dropout: Embedding层的dropout. 若为None或1, 则无该Dropout层.
        :param embedding_kwargs: Embedding层的其他参数, 参考 keras.layers.embeddings.Embedding
        :param lstm_units: LSTM层的神经元数
        :param lstm_dropout: LSTM层的dropout. 若为None或1, 则无该Dropout层.
        :param lstm_kwargs: LSTM层的其他参数, 参考 keras.layers.LSTM
        :param dense: 定义每个全连接层的名称, 神经元数, 参数初始化函数, 激活函数, dropout等. 若为None或[], 则无全连接层
        """

        self.name = name

        if embedding_kwargs is not None:
            assert 'input_dim' not in embedding_kwargs
            assert 'output_dim' not in embedding_kwargs
            assert 'input_length' not in embedding_kwargs
            self.embeddings_kwargs = embedding_kwargs
        else:
            self.embeddings_kwargs = {}

        self.embedding_name = 'embedding'
        if 'name' in self.embeddings_kwargs:
            self.embedding_name = self.embeddings_kwargs.pop('name')

        self.embedding_word_number = embedding_word_number
        self.embedding_vector_length = embedding_vector_length
        assert embedding_dropout is None or (0 < embedding_dropout < 1)
        self.embedding_dropout = embedding_dropout

        if lstm_kwargs is not None:
            assert 'units' not in lstm_kwargs
            self.lstm_kwargs = lstm_kwargs
        else:
            self.lstm_kwargs = {}

        self.lstm_name = 'lstm'
        if 'name' in self.lstm_kwargs:
            self.lstm_name = self.lstm_kwargs.pop('name')
        self.lstm_units = lstm_units
        assert lstm_dropout is None or (0 < lstm_dropout < 1)
        self.lstm_dropout = lstm_dropout

        if dense is None:
            dense = []
        else:
            for layer in dense:
                assert isinstance(layer, dict)
                assert 'units' in layer and layer['units'] is not None and type(layer['units']) is int
                assert 'activation' in layer and layer['activation'] is not None and type(layer['activation']) is str
                if 'dropout' in layer:
                    assert layer['dropout'] is None or (type(layer['dropout']) is float and (0 < layer['dropout'] < 1))
        self.dense = dense

    def __call__(self, inputs, **kwargs):
        """
        模型调用方法, 实现函数式运算
        :param inputs:
        :param kwargs:
        :return:
        """

        embedding_layer_name = self.name + '.' + self.embedding_name
        x_embedded = Embedding(name=embedding_layer_name,
                               input_dim=self.embedding_word_number,
                               output_dim=self.embedding_vector_length,
                               **self.embeddings_kwargs
                               )(inputs)

        if self.embedding_dropout is not None:
            x_embedded = Dropout(self.embedding_dropout, name=embedding_layer_name+'.dropout')(x_embedded)

        lstm_layer_name = self.name + '.' + self.lstm_name
        x_lstm = LSTM(name=lstm_layer_name, units=self.lstm_units, **kwargs)(x_embedded)

        if self.lstm_dropout is not None:
            x_lstm = Dropout(self.lstm_dropout, name=lstm_layer_name + '.dropout')(x_lstm)

        x_dense = x_lstm
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
                x_dense = Dropout(dropout, name=name+'.dropout')(x_dense)

        return x_dense


if __name__ == '__main__':

    x = Input(shape=(128,), dtype='int32', name='x')

    y = EmbeddingLSTM(name='ELSTM',

                      embedding_word_number=500,
                      embedding_vector_length=100,
                      embedding_dropout=0.8,
                      embedding_kwargs={'name': 'e', 'embeddings_initializer': 'lecun_uniform'},

                      lstm_units=50,
                      lstm_dropout=0.8,
                      lstm_kwargs={'name': 'l'},

                      dense=[
                          {'units': 20, 'activation': 'tanh', 'dropout': 0.9, 'name': 'd'},
                          {'units': 1, 'activation': 'sigmoid', 'dropout': 0.9, 'name': 'd'}
                      ]

                      )(x)
    model = Model(inputs=[x], outputs=[y])
    model.summary()

