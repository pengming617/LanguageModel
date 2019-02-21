import gensim
# 导入模型
model = gensim.models.KeyedVectors.load_word2vec_format('word2vec/vectors.bin', binary=True)

# 返回一个词 的向量：
print(model['word'])