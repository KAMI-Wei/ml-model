# 机器学习模型

本项目维护一些自行实现或复现的模型

## keras

基于 keras 实现的模型, 实现为与 keras.layer 同构, 维护在目录 keras 下.

### EmbeddingLSTM

经典 Embedding, LSTM, Dense(s) 三层模型


## NLTK

基于 NLTK 实现的模型, 维护在在目录 n/model 下

### NgramModel

在NLTK 3.0版本后, 因为接口有较大调整, 子包'model'的迁移出现问题, 被维护者暂时移除但迟迟没有合并回去. 对应地, 维护者在'model'分支上提供了Ngram模型的基类'BaseNgramModel', 要求使用者通过这个基类实现自己的模型. 本人根据此基类, 实现递归NgramCounter, 并重新实现2.x版本的Katz backoff平滑Ngrams模型.

- n/model/model.py: 目前nltk的'model'分支上的'BaseNgramModel'基类
- n/model/ngram.py: 继承'BaseNgramModel'实现的Katz backoff平滑Ngrams模型
- n/model/legacy/ngram.py: nltk 2.x版本的Katz backoff平滑Ngrams模型