import numpy as np
import tensorflow as tf
import config as config
import lstm_lm as lstm_lm

con = config.Config()


# 使用给定的模型model在数据data上运行train_op并返回再全部数据上的perplexity值
def run_epoch(session, model, batches, train_op, output_log, step):
    # 计算平均perplexity的辅助变量
    total_costs = 0.0
    iters = 0
    # 训练一个epoch
    for x, y in batches:
        # 在当前batch上运行train_op并计算损失值，交叉熵损失函数计算的就是下一个单词为给定单词的概率
        cost, _ = session.run(
            [model.cost, train_op],
            {model.input_data: x, model.targets: y, model.sequence_length: np.array([model.num_steps] * model.batch_size)}
        )
        total_costs += cost
        iters += model.num_steps
        # 只有在训练时输出日志
        if output_log and step % 1000 == 0:
            print('After %d steps, perplexity is %.3f' % (step, np.exp(total_costs / iters)))
        step += 1
    # 返回给定模型在给定数据上的perplexity值
    return step, np.exp(total_costs / iters)


# 从文件中读取数据，并返回包含单词编号的数组
def read_data(file_path):
    with open(file_path, 'r') as fin:
        # 将整个文档读进一个长字符串
        lines = []
        for x in fin.readlines():
            lines.append(x)
        id_string = ' '.join([line.strip() for line in lines])
    id_list = [int(w) for w in id_string.split()]  # 将读取的单词编号转为整数
    print(file_path + " read success")
    return id_list


def make_batch(id_list, batch_size, num_step):
    # 计算总的batch数量，每个batch包含的单词数量是batch_size * num_step
    num_batches = (len(id_list) - 1) // (batch_size * num_step)
    # 将数据整理成一个维度为[batch_size, num_batches * num_step]的二维数组
    data = np.array(id_list[: num_batches * batch_size * num_step])
    data = np.reshape(data, [batch_size, num_batches * num_step])
    # 沿着第二个维度将数据切分成num_batches个batch,存入一个数组
    data_batches = np.split(data, num_batches, axis=1)

    # 重复上述操作，但是每个位置向右移动一位，这里得到的时RNN每一步输出所需要预测的下一个单词
    label = np.array(id_list[1: num_batches * batch_size * num_step + 1])
    label = np.reshape(label, [batch_size, num_batches * num_step])
    label_batches = np.split(label, num_batches, axis=1)
    # 返回一个长度为num_batches的数组，其中每一项包含一个data矩阵和一个label矩阵
    return list(zip(data_batches, label_batches))


def main():
    # 定义初始化函数
    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    # 定义训练用的循环神经网络模型
    with tf.variable_scope('language_model', reuse=None, initializer=initializer):
        train_model = lstm_lm.Lstm_LanguageModel(True, con.TRAIN_BATCH_SIZE, con.TRAIN_NUM_STEP, con.VOCAB_SIZE,
                                                 con.HIDDEN_SIZE, con.NUM_LAYERS, con.LSTM_KEEP_PROB)
    # 定义测试用的循环神经网络模型。它与train_model公用参数，但是没有dropout
    with tf.variable_scope('language_model', reuse=True, initializer=initializer):
        eval_model = lstm_lm.Lstm_LanguageModel(False, con.EVAL_BATCH_SIZE, con.EVAL_NUM_STEP, con.VOCAB_SIZE,
                                                con.HIDDEN_SIZE, con.NUM_LAYERS, 1.0)
    # 训练模型
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        train_batches = make_batch(read_data(con.TRAIN_DATA), con.TRAIN_BATCH_SIZE, con.TRAIN_NUM_STEP)
        eval_batches = make_batch(read_data(con.EVAL_DATA), con.EVAL_BATCH_SIZE, con.EVAL_NUM_STEP)
        test_batches = make_batch(read_data(con.TEST_DATA), con.EVAL_BATCH_SIZE, con.EVAL_NUM_STEP)

        step = 0
        min_perplexity = 999999.0
        for i in range(con.NUM_EPOCH):
            print('In iteration: %d' % (i + 1))
            step, train_pplx = run_epoch(sess, train_model, train_batches, train_model.train_op, True, step)
            print('Epoch: %d Train Perplexity: %.3f' % (i + 1, train_pplx))
            _, eval_pplx = run_epoch(sess, eval_model, eval_batches, tf.no_op(), False, 0)
            print('Epoch: %d Eval Perplexity: %.3f' % (i + 1, eval_pplx))
            if eval_pplx < min_perplexity:
                min_perplexity = eval_pplx
                saver.save(sess, "model/lstm_lm.ckpt")

        _, test_pplx = run_epoch(sess, eval_model, test_batches, tf.no_op(), False, 0)
        print('Test Perplexity: %.3f' % test_pplx)


if __name__ == '__main__':
    main()
