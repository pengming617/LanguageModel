# LanguageModel
基于LSTM Bi-lstm的语言模型
语料为中文维基百科，wiki_train, wiki_valid, wiki_test
wiki_vocab 是对语料进行结巴分词后构成的词典
语料中每个词语已经替换为对应的id

train.py 训练语言模型
compute_sentence_pro 利用语言模型计算一句话的概率