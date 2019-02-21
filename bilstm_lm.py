import tensorflow as tf


class Lstm_LanguageModel(object):

    def __init__(self, is_training, num_steps, VOCAB_SIZE, HIDDEN_SIZE, NUM_LAYERS, dropout_keep_prob):

        # 定义每一步的输出和预期输出，两个的维度都是[batch_size, num_steps]
        self.input_data = tf.placeholder(tf.int32, [None, num_steps], name='input_x')
        self.targets = tf.placeholder(tf.int32, [None, num_steps], name='input_y')
        self.sequence_length = tf.placeholder(tf.int32, [None], name='sequence_length')

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
        fc_w = tf.Variable(tf.truncated_normal([HIDDEN_SIZE * 2, VOCAB_SIZE], stddev=0.1), name='fc_w')
        fc_b = tf.Variable(tf.zeros([VOCAB_SIZE]), name='fc_b')
        outputs = tf.reshape(outputs, [-1, HIDDEN_SIZE * 2])

        self.logits = tf.matmul(outputs, fc_w) + fc_b

        # 定义交叉熵损失函数和平均损失
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self.targets, [-1]), logits=self.logits)
        self.cost = tf.reduce_sum(loss)

        # 只在训练模型时定义反向传播操作
        if not is_training:
            return

        # 梯度更新
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)
        optimizer = tf.train.AdamOptimizer(0.001)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))


if __name__ == '__main__':
    Lstm_LanguageModel(True, 20, 10000, 200, 2, 0.5)