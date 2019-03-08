import re
import tensorflow as tf
import numpy as np
import jieba
import config as config

con = config.Config()


class Compute_pro(object):

    def __init__(self):
        # 加载词典
        file = open('model/wiki.vocab', 'r')
        self.word2id = {}
        i = 0
        for line in file.readlines():
            self.word2id[line.strip()] = i
            i += 1
        self.checkpoint_file = tf.train.latest_checkpoint('model/Bilstm_model')
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            self.sess = tf.Session(config=session_conf)
            with self.sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(self.checkpoint_file))
                saver.restore(self.sess, self.checkpoint_file)

                # Get the placeholders from the graph by name
                self.input_x = graph.get_operation_by_name("language_model/input_x").outputs[0]
                self.seq_length = graph.get_operation_by_name("language_model/sequence_length").outputs[0]
                self.drop_keep_prob = graph.get_operation_by_name("language_model/drop_out_keep").outputs[0]

                # Tensors we want to evaluate
                self.predictions = graph.get_operation_by_name("language_model/prediction").outputs[0]

    def compute_sentence_pro(self, sentence):
        text = sentence.replace("\n", "")
        # 删除（）里的内容
        text = re.sub('（[^（.]*）', '', text)
        # 只保留中文部分
        text = ''.join([x for x in text if '\u4e00' <= x <= '\u9fa5'])
        # 利用jieba进行分词
        words = ' '.join(jieba.cut(text)).split(" ")
        # 在开头插上<EOS>
        word2ids = [1]
        for word in words:
            if word not in self.word2id.keys():
                word2ids.append(0)
            else:
                word2ids.append(self.word2id[word])

        if len(word2ids) < con.NUM_STEP:
            word2ids[len(word2ids): con.NUM_STEP] = [0] * (con.NUM_STEP - len(word2ids))

        feed_dict = {
            self.input_x: np.array([word2ids]),
            self.seq_length: np.array([len(words)]),
            self.drop_keep_prob: 1.0
        }
        y = self.sess.run([self.predictions], feed_dict)
        log_p_sentence = 0.0
        for i in range(len(words)):
            prob_word = y[0][i][word2ids[i+1]]
            log_p_sentence += np.log(prob_word)
        pro = np.exp(log_p_sentence)
        return pro


if __name__ == '__main__':
    cp = Compute_pro()
    sen = "范德萨佛挡杀佛发打发斯蒂芬"
    y = cp.compute_sentence_pro(sen)
    y = '{:.8f}'.format(y)
    print(sen + "，这句话出现的概率为:"+str(y))