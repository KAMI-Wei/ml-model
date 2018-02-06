"""
Bayes相关模型
"""
from ngram import NgramCounter


class BayesCorrector(object):

    def __init__(self, counter):
        self._counter = counter

    # 返回所有编辑距离为1的单词集合
    def edits1(self, word):
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        n = len(word)
        return set([word[0:i] + word[i + 1:] for i in range(n)] +  # deletion
                   [word[0:i] + word[i + 1] + word[i] + word[i + 2:] for i in range(n - 1)] +  # transposition
                   [word[0:i] + c + word[i + 1:] for i in range(n) for c in alphabet] +  # alteration
                   [word[0:i] + c + word[i:] for i in range(n + 1) for c in alphabet])  # insertion

    def known_edits2(self, word):
        """
        返回所有编辑距离为2的单词集合
        :param word:
        :return:
        """
        return set(e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

    def known(self, words):
        """
        只把那些正确的词作为候选词
        :param words:
        :return:
        """
        return set(w for w in words if w in self._counter)

    def correct(self, word):
        """
        如果known(set)非空, candidate 就会选取这个集合, 而不继续计算后面的
        :param word:
        :return:
        """
        candidates = self.known([word]) or self.known(self.edits1(word)) or self.known_edits2(word) or [word]
        print(candidates)
        return max(candidates, key=lambda w: self._counter[w])


if __name__ == '__main__':
    import nltk.book as book
    import json

    ngramCounter = NgramCounter(order=2, train=book.text1)

    print(ngramCounter.vocabulary)

    counter = book.text1.vocab()

    corrector = BayesCorrector(counter)

    print(json.dumps(corrector.correct("helllo"), indent=2))

