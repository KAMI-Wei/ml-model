"""
Bayes相关模型
"""
import pickle
from collections import defaultdict
from ngram import NgramCounter, NgramModel


class BayesCorrector(object):

    def __init__(self, ngram_model: NgramModel, tokenizer=None):
        self._ngram_model = ngram_model

        self._lookup_table = defaultdict(lambda: 0)

        self._dictionary = set()

        for word in self._ngram_model.ngram_counter.counter[1].vocabulary:
            self._dictionary.add(word[0])

    # 返回所有编辑距离为1的单词集合
    def edits1(self, word):
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        n = len(word)
        return set([word[0:i] + word[i + 1:] for i in range(n)] +  # deletion
                   [word[0:i] + word[i + 1] + word[i] + word[i + 2:] for i in range(n - 1)] +  # transposition
                   [word[0:i] + c + word[i + 1:] for i in range(n) for c in alphabet] +  # alteration
                   [word[0:i] + c + word[i:] for i in range(n + 1) for c in alphabet])  # insertion
    #

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
        return set(w for w in words if w in self._dictionary)

    def correct(self, text):
        text_corrected = []

        for ngram in self._ngram_model.ngram_counter.to_ngrams(text):

            word = ngram[-1]
            context = ngram[:-1]

            # TODO 句法分析
            if word in self._dictionary:
                text_corrected.append(word)
                continue

            candidates = self.known(self.edits1(word))
            if len(candidates) == 0:
                text_corrected.append(word)
                continue

            if ngram in self._lookup_table:
                text_corrected.append(self._lookup_table[ngram])
                continue

            best_candidate = None
            best_prob = 0
            for candidate in candidates:
                prob = self._ngram_model.score(candidate, context)
                if prob > best_prob:
                    best_candidate = candidate
                    best_prob = prob
            self._lookup_table[ngram] = best_candidate
            text_corrected.append(best_candidate)

        return text_corrected

    def save(self):
        return pickle.dumps(obj=self._ngram_model, protocol=True, fix_imports=True)




if __name__ == '__main__':
    from nltk.text import Text
    from nltk.corpus import gutenberg
    text1 = Text(gutenberg.words('melville-moby_dick.txt'))
    import json

    ngramCounter = NgramCounter(order=2, train=text1)
    ngramModel = NgramModel(ngram_counter=ngramCounter)

    corrector = BayesCorrector(ngram_model=ngramModel)
    print(corrector.correct(['I', 'don', 'think', 'you', 'rre', 'good']))
    print(corrector.save())


