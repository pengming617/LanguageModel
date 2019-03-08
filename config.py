class Config(object):

    def __init__(self):

        self.TRAIN_DATA = 'data/wiki_train.txt'  # 训练数据路径
        self.EVAL_DATA = 'data/wiki_valid.txt'  # 验证数据路径
        self.TEST_DATA = 'data/wiki_test.txt'  # 测试数据路径
        self.VOCAB_SIZE = 526304  # 词典规模
        self.HIDDEN_SIZE = 200  # 隐藏层规模
        self.NUM_LAYERS = 1  # 深层循环神经网络中LSTM结构的层数
        self.BATCH_SIZE = 32  # 训练数据batch的大小
        self.NUM_STEP = 30  # 训练数据截断长度
        self.num_sampled = 10000  # 采样大小

        self.NUM_EPOCH = 2  # 使用训练数据的轮数
        self.LSTM_KEEP_PROB = 0.5  # LSTM节点不被dropout的概率
        self.MAX_GRAD_NORM = 5  # 用于控制梯度膨胀的梯度大小上限
        self.SHARE_EMB_AND_SOFTMAX = True  #在softmax层和词向量层之间共享参数