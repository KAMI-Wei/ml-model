"""
贝叶斯拼写检查
"""

import re
import collections


# 把语料中的单词全部抽取出来, 转成小写, 并且去除单词中间的特殊符号
def words(text):
    new_words = re.findall('[a-z]+', text.lower())
    return new_words


# 统计每个单词出现的次数
def train(features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] = model[f] + 1
    return model


# 所有单词以及出现的次数
NWORDS = train(words(open('big.txt').read()))


# 返回所有编辑距离为1的单词集合
def edits1(word):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    n = len(word)
    return set([word[0:i] + word[i + 1:] for i in range(n)] +  # deletion
               [word[0:i] + word[i + 1] + word[i] + word[i + 2:] for i in range(n - 1)] +  # transposition
               [word[0:i] + c + word[i + 1:] for i in range(n) for c in alphabet] +  # alteration
               [word[0:i] + c + word[i:] for i in range(n + 1) for c in alphabet])  # insertion


# 返回所有编辑距离为2的单词集合
def known_edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1))


# 只把那些正确的词作为候选词
def known(words):
    return set(w for w in words if w in NWORDS)


# 如果known(set)非空, candidate 就会选取这个集合, 而不继续计算后面的
def correct(word):
    candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
    print(candidates)
    return max(candidates, key=lambda w: NWORDS[w])


print(correct('namg'))