"""
拼写纠正器

"""

import pickle
from collections import defaultdict
from ngram import NgramCounter, NgramModel
from itertools import product

vowels = set('aeiouy')
alphabet = set('abcdefghijklmnopqrstuvwxyz')


# ** UTILITY FUNCTIONS **


def number_of_dupes(string, idx):
    """返回string[idx]在后面被重复的次数"""
    # "abccdefgh", 2  returns 1
    initial_idx = idx
    last = string[idx]
    while idx + 1 < len(string) and string[idx + 1] == last:
        idx += 1
    return idx - initial_idx


# ** POSSIBILITIES ANALYSIS **


def variants(word):
    """获取所有编辑距离为1的词/词组"""
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
    replaces = [a + c + b[1:] for a, b in splits for c in alphabet if b]
    inserts = [a + c + b for a, b in splits for c in alphabet]
    return set(deletes + transposes + replaces + inserts)


def double_variants(word):
    """获取所有编辑距离为2的词组"""
    return set(s for w in variants(word) for s in variants(w))


def reductions(word):
    """获取删除重复字母后的单词"""
    word = list(word)
    # ['h','i', 'i', 'i'] becomes ['h', ['i', 'ii', 'iii']]
    for idx, l in enumerate(word):
        n = number_of_dupes(word, idx)
        # if letter appears more than once in a row
        if n:
            # generate a flat list of options ('hhh' becomes ['h','hh','hhh'])
            # only take up to 3, there are no 4 letter repetitions in english
            flat_dupes = [l * (r + 1) for r in range(n + 1)][:3]
            # remove duplicate letters in original word
            for _ in range(n):
                word.pop(idx + 1)
            # replace original letter with flat list
            word[idx] = flat_dupes

    # ['h',['i','ii','iii']] becomes 'hi','hii','hiii'
    for p in product(*word):
        yield ''.join(p)


def vowel_swaps(word):
    """获取元音替换后的单词"""
    word = list(word)
    # ['h','i'] becomes ['h', ['a', 'e', 'i', 'o', 'u', 'y']]
    for idx, l in enumerate(word):
        if type(l) == list:
            pass  # dont mess with the reductions
        elif l in vowels:
            word[idx] = list(vowels)  # if l is a vowel, replace with all possible vowels

    # ['h',['i','ii','iii']] becomes 'hi','hii','hiii'
    for p in product(*word):
        yield ''.join(p)


def both(word):
    """获取删除重复字母和元音替换后的单词"""
    for reduction in reductions(word):
        for variant in vowel_swaps(reduction):
            yield variant


# ** POSSIBILITY CHOOSING **

def suggestions(word, real_words, short_circuit=True):
    """返回修改建议. 若short_circuit = True, 返回第一个吻合的最佳建议, 否则返回所有建议"""
    word = word.lower()
    if short_circuit:
        return ({word} & real_words or                      # caps     "inSIDE" => "inside"
                set(reductions(word)) & real_words or       # repeats  "jjoobbb" => "job"
                set(vowel_swaps(word)) & real_words or      # vowels   "weke" => "wake"
                set(variants(word)) & real_words or         # other    "nonster" => "monster"
                set(both(word)) & real_words or             # both     "CUNsperrICY" => "conspiracy"
                set(double_variants(word)) & real_words or  # other    "nmnster" => "manster"
                {"NO SUGGESTION"})
    else:
        return ({word} & real_words or
                (set(reductions(word)) | set(vowel_swaps(word)) | set(variants(word)) | set(both(word)) | set(
                    double_variants(word))) & real_words or {"NO SUGGESTION"})


class NgramCorrector(object):

    def __init__(self, ngram_model: NgramModel, lookup_table: dict=None):
        self._ngram_model = ngram_model

        self._lookup_table = defaultdict(lambda: 0)
        if lookup_table is not None:
            self._lookup_table.update(lookup_table)

        self._dictionary = set()

        for word in self._ngram_model.ngram_counter.counter[1].vocabulary:
            self._dictionary.add(word[0])

        self._ngram_model_order = self._ngram_model.ngram_counter.order

    def correct(self, text: list):
        text_corrected = []

        corrected_word = ''
        corrected_word_ctx_idx = -1
        for ngram in self._ngram_model.ngram_counter.to_ngrams(text):

            # 如果前面已经有修改过的词, 那么往后的上下文要修改
            if corrected_word_ctx_idx >= 0:
                ngram = ngram[0: corrected_word_ctx_idx] + (corrected_word,) + ngram[corrected_word_ctx_idx+1:]
                corrected_word_ctx_idx -= 1

            word = ngram[-1]
            context = ngram[:-1]

            # 如果这个词在词典里, 那么就不是拼写错误
            # TODO 句法分析
            if word in self._dictionary:
                text_corrected.append(word)
                continue


            # 如果已经修正过这个词, 那就用之前修正过的
            if ngram in self._lookup_table:
                corrected_word = self._lookup_table[ngram]
                corrected_word_ctx_idx = self._ngram_model_order - 1 - 1
                text_corrected.append(corrected_word)
                continue

            # 获取所有候选单词
            candidates = suggestions(word, self._dictionary, short_circuit=False)

            # 如果没有候选单词, 消极处理
            if len(candidates) == 0:
                text_corrected.append(word)
                continue

            # 计算所有ngram的概率, 选一个概率最大的
            best_candidate = None
            best_prob = 0
            for candidate in candidates:
                prob = self._ngram_model.score(candidate, context)
                if prob > best_prob:
                    best_candidate = candidate
                    best_prob = prob
            self._lookup_table[ngram] = best_candidate
            text_corrected.append(best_candidate)
            corrected_word = best_candidate
            corrected_word_ctx_idx = self._ngram_model_order - 1 - 1

        return text_corrected

    @property
    def lookup_table(self):
        return self._lookup_table

    def save(self, path):
        """
        保存模型
        :param path: 保存路径
        :return:
        """
        params_dict = {
            '_ngram_model_pickle': pickle.dumps(obj=self._ngram_model, protocol=True, fix_imports=True),
            '_lookup_table': dict(self._lookup_table),
        }
        with open(file=path, mode="wb") as fp:
            fp.write(pickle.dumps(obj=params_dict))

    @classmethod
    def load(cls, path):
        """
        加载模型
        :param path: 保存路径
        :return:
        """
        params_dict = pickle.load(open(file=path, mode="rb"))
        lookup_table = params_dict['_lookup_table']
        ngram_model = pickle.loads(params_dict['_ngram_model_pickle'], fix_imports=True)
        return cls(ngram_model=ngram_model, lookup_table=lookup_table)


if __name__ == '__main__':
    from nltk.text import Text
    from nltk.corpus import gutenberg

    text1 = Text(gutenberg.words('melville-moby_dick.txt'))
    #
    ngramCounter = NgramCounter(order=2, train=text1)
    ngramModel = NgramModel(ngram_counter=ngramCounter)

    corrector = NgramCorrector(ngram_model=ngramModel)
    print(corrector.correct(['I', 'dooo', 'think', 'you', 'rre', 'goooood']))
    corrector2 = NgramCorrector.load("123")
    print(corrector2.correct(['I', 'don', 'think', 'you', 'rre', 'goooood']))
