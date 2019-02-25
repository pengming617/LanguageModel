import tensorflow as tf
import config as config

con = config.Config()


class Lstm_LanguageModel(object):

    def __init__(self, is_training, batch_size, num_steps, VOCAB_SIZE, HIDDEN_SIZE, NUM_LAYERS, dropout_keep_prob):

        # 定义每一步的输出和预期输出，两个的维度都是[batch_size, num_steps]
        self.input_data = tf.placeholder(tf.int32, [None, num_steps], name='input_x')
        self.targets = tf.placeholder(tf.int32, [None, num_steps], name='input_y')
        self.sequence_length = tf.placeholder(tf.int32, [None], name='sequence_length')
        self.num_steps = num_steps
        self.batch_size = batch_size

        # embedding layer
        with tf.device('/gpu:0'), tf.name_scope("embedding"):
            embedding = tf.get_variable('embedding', [VOCAB_SIZE, HIDDEN_SIZE])
            # 将输入单词转换为词向量
            inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        # build model
        fw_lstm_cells = [
            tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE), output_keep_prob=dropout_keep_prob) for _ in range(NUM_LAYERS)
        ]
        fw_cell = tf.nn.rnn_cell.MultiRNNCell(fw_lstm_cells)

        bw_lstm_cells = [
            tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE), output_keep_prob=dropout_keep_prob) for _ in range(NUM_LAYERS)
        ]
        bw_cell = tf.nn.rnn_cell.MultiRNNCell(bw_lstm_cells)

        (outputs, output_states) = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs,
                                                                   sequence_length=self.sequence_length,
                                                                   dtype=tf.float32)
        outputs = tf.concat(outputs, axis=2)

        # outputs shape
        weight = tf.Variable(tf.truncated_normal([HIDDEN_SIZE, VOCAB_SIZE], stddev=0.1), name='fc_w')
        bias = tf.Variable(tf.zeros([VOCAB_SIZE]), name='fc_b')
        outputs = tf.reshape(outputs, [-1, HIDDEN_SIZE])

        self.logits = tf.matmul(outputs, weight) + bias
        self.prediction = tf.nn.softmax(self.logits, name='prediction')

        if is_training:
            # 采用tf.nn.sampled_softmax_loss 加快模型的计算速度
            loss = tf.nn.sampled_softmax_loss(tf.transpose(weight),
                                              bias,
                                              tf.reshape(self.targets, [-1, 1]),
                                              outputs,
                                              con.num_sampled,
                                              VOCAB_SIZE,
                                              partition_strategy="div",
                                              name="sampled_softmax_loss",
                                              seed=None)
        else:
            # 定义交叉熵损失函数和平均损失
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.reshape(self.targets, [-1]),
                logits=self.logits
            )
        self.cost = tf.reduce_sum(loss) / (batch_size * num_steps)

        # 只在训练模型时定义反向传播操作
        if not is_training:
            return

        # 梯度更新
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)
        optimizer = tf.train.AdamOptimizer(0.001)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
