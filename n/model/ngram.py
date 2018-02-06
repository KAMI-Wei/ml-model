from __future__ import unicode_literals, division

from nltk import compat, ngrams
from nltk.probability import ConditionalProbDist, ConditionalFreqDist, LidstoneProbDist

from model import BaseNgramModel


def _estimator(fdist, *estimator_args, **estimator_kwargs):
    """
    Default estimator function using a LidstoneProbDist.
    """
    # can't be an instance method of NgramModel as they
    # can't be pickled either.
    return LidstoneProbDist(fdist, *estimator_args, **estimator_kwargs)


class NgramCounter(object):
    """
    依据 NLTK 3.0 给出的模型基类'BaseNgramModel'所实现的NgramCounter

    必要成员属性和方法
    - order: 属性, int, 模型阶数
    - ngrams: 属性, dict<int, ConditionalFreqDist>, 各阶模型的条件概率分布的集合
    - vocabulary: 属性, set<tuple<str>>, ngram词汇表
    - to_gram: 方法, (list<str>)-> yield tuple<str>, 通过输入文本生成ngram
    - check_against_vocab: 方法, (str)-> str, 根据词汇表对单词做映射

    """
    def __init__(self, order: int, train: list,
                 pad_left: bool=True, pad_right: bool =False, left_pad_symbol: str ='', right_pad_symbol: str ='',
                 recursive: bool =True):
        """

        :param order: 模型阶数
        :param train: 训练样本
        :param pad_left: 是否进行左填充
        :param pad_right: 是否进行右填充
        :param left_pad_symbol: 左填充符号
        :param right_pad_symbol: 右填充符号
        :param recursive: 是否生成低阶模型
        """
        self._ngrams = dict()
        self._counter = dict()

        # 模型阶数必须大于0
        assert (order > 0), order
        # 保存模型阶数
        self._order = order
        # 为方便检查, 为n=1的1阶模型保存一个快捷变量

        # padding的设置
        assert (isinstance(pad_left, bool))
        assert (isinstance(pad_right, bool))
        self._pad_left = pad_left
        self._pad_right = pad_right
        self._left_pad_symbol = left_pad_symbol
        self._right_pad_symbol = right_pad_symbol

        cfd = ConditionalFreqDist()
        self._vocabulary = set()

        # 输入适配. 如果输入的训练数据不是list<list<str>>, 用一个列表包裹它
        if (train is not None) and isinstance(train[0], compat.string_types):
            train = [train]

        for sent in train:
            for ngram in self.to_ngrams(sent):
                self._vocabulary.add(ngram)
                context = tuple(ngram[:-1])
                token = ngram[-1]
                # NB, ConditionalFreqDist的接口已经改变, 已经没有方法'inc', 需要改为如下语句
                cfd[context][token] += 1

        self._ngrams[self._order] = cfd
        self._counter[self._order] = self

        # NB, 关键代码: 递归生成低阶NgramCounter
        # 如果递归, 那就生成低阶概率分布, 注意还要把order-2至1阶的概率分布取回来
        if recursive and not order == 1:
            self._backoff = NgramCounter(order - 1, train,
                                         pad_left=pad_left, left_pad_symbol=left_pad_symbol,
                                         pad_right=pad_right, right_pad_symbol=right_pad_symbol)
            # 递归地把个低阶概率分布取回来
            cursor = self._backoff
            while cursor is not None:
                self._ngrams[cursor.order] = cursor.ngrams[cursor.order]
                self._counter[cursor.order] = cursor
                cursor = cursor.backoff
        else:
            self._backoff = None

    @property
    def order(self) -> int:
        return self._order

    @property
    def vocabulary(self) -> set:
        return self._vocabulary

    @property
    def ngrams(self) -> dict:
        return self._ngrams

    @property
    def counter(self) -> dict:
        return self._counter

    @property
    def backoff(self) -> type('NgramCounter'):
        return self._backoff

    def check_against_vocab(self, word) -> str:
        """
        目前不对生词作任何处理
        :param word:
        """
        return word

    def to_ngrams(self, text) -> tuple:
        return ngrams(text, self._order,
                      pad_left=self._pad_left, pad_right=self._pad_right,
                      left_pad_symbol=self._left_pad_symbol, right_pad_symbol=self._right_pad_symbol)


class NgramModel(BaseNgramModel):
    """
    继承模型基类'BaseNgramModel'重新实现NgramModel

    Note:
        1. 原方法'prob'和'logprob'已分别改名为'score'和'logstore'
        2. 原方法'entropy'显式对输入文本进行padding, 然而基类'BaseNgramModel'的'entorpy'没有.
            但是, 基类'BaseNgramModel'的'entorpy'的调用'NgramCounter'to_ngram, 已经进行padding.
            所以我们不需要覆盖'entropy'
    """

    def __init__(self, ngram_counter, estimator=None, *estimator_args, **estimator_kwargs):

        super(NgramModel, self).__init__(ngram_counter)

        # 设置频率平滑器, 没有就使用默认
        if estimator is None:
            estimator = _estimator

        # 使用频率平滑器, 生成ngram模型
        if not estimator_args and not estimator_kwargs:
            self._model = ConditionalProbDist(self.ngrams, estimator, len(self.ngrams))
        else:
            self._model = ConditionalProbDist(self.ngrams, estimator, *estimator_args, **estimator_kwargs)

        # 递归生成低阶模型
        if self._order > 1 and self.ngram_counter.backoff is not None:
            self._backoff = NgramModel(self.ngram_counter.backoff, estimator, *estimator_args, **estimator_kwargs)

    def score(self, word, context):
        """
        Evaluate the probability of this word in this context using Katz Backoff.
        :param word: the word to get the probability of
        :type word: str
        :param context: the context the word is in
        :type context: list(str)
        """

        context = tuple(context)
        # NB, 属性'_ngrams'已经在基类'BaseNgramModel'被赋值为'NgramCounter'的ConditionalFreqDist集合.
        # 词汇表实际上是NgramCounter的属性'vocabulary'. 具体修改如下
        # if (context + (word,) in self._ngrams) or (self._n == 1):
        if (context + (word,) in self.ngram_counter.vocabulary) or (self._order == 1):
            return self[context].prob(word)
        else:
            return self._alpha(context) * self._backoff.score(word, context[1:])

    def _alpha(self, tokens):
        return self._beta(tokens) / self._backoff._beta(tokens[1:])

    def _beta(self, tokens):
        # TODO 有些 estimator 没有 discount, 这个时候得改用 prob
        return self[tokens].discount() if tokens in self else 1

    def choose_random_word(self, context):
        """
        Randomly select a word that is likely to appear in this context.
        :param context: the context the word is in
        :type context: list(str)
        """

        return self.generate(1, context)[-1]

    # NB, this will always start with same word if the model
    # was trained on a single text
    def generate(self, num_words, context=()):
        """
        Generate random text based on the language model.
        :param num_words: number of words to generate
        :type num_words: int
        :param context: initial words in generated string
        :type context: list(str)
        """

        text = list(context)
        for i in range(num_words):
            text.append(self._generate_one(text))
        return text

    def _generate_one(self, context):
        context = (self._lpad + tuple(context))[-self._n + 1:]
        if context in self:
            return self[context].generate()
        elif self._n > 1:
            return self._backoff._generate_one(context[1:])
        else:
            return '.'

    def __contains__(self, item):
        return tuple(item) in self._model

    def __getitem__(self, item):
        return self._model[tuple(item)]

    def __repr__(self):
        return '<NgramModel with %d %d-grams>' % (len(self._ngrams), self._n)


__all__ = ['NgramCounter', 'NgramModel', ]

if __name__ == '__main__':
    from nltk.book import text6
    counter = NgramCounter(order=2, train=text6, pad_left=True, left_pad_symbol='')
    m = NgramModel(ngram_counter=counter, gamma=1, bins=None)
    print(m.score("Journal", ("Street",)))
    print(m.score("think", ("I",)))
    print(m.score("don't", ("I",)))
    print(m.score("way", ("Go",)))
    print(m.entropy(['I', 'hate', 'you']))
