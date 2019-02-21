import tensorflow as tf
import numpy as np

writer = tf.python_io.TFRecordWriter('test.tfrecord')

for i in range(0, 2):
    a = np.random.random(size=(180)).astype(np.float32)
    a = a.data.tolist()
    b = [2016 + i, 2017+i]
    c = np.array([[0, 1, 2],[3, 4, 5]]) + i
    c = c.astype(np.uint8)
    c_raw = c.tostring()#这里是把ｃ换了一种格式存储
    print('  i:', i)
    print('  a:', a)
    print('  b:', b)
    print('  c:', c)
    example = tf.train.Example(features=tf.train.Features(
            feature = {'a':tf.train.Feature(float_list = tf.train.FloatList(value=a)),
                       'b':tf.train.Feature(int64_list = tf.train.Int64List(value = b)),
                       'c':tf.train.Feature(bytes_list = tf.train.BytesList(value = [c_raw]))}))
    serialized = example.SerializeToString()
    writer.write(serialized)
    print('   writer',i,'DOWN!')
writer.close()


filename_queue = tf.train.string_input_producer(['test.tfrecord'], num_epochs=None)
# create a reader from file queue
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
# get feature from serialized example

features = tf.parse_single_example(serialized_example,
        features={
            'a': tf.FixedLenFeature([180], tf.float32),
            'b': tf.FixedLenFeature([2], tf.int64),
            'c': tf.FixedLenFeature([],tf.string)
        }
    )
a_out = features['a']
b_out = features['b']
c_out = features['c']
#c_raw_out = features['c']
#c_raw_out = tf.sparse_to_dense(features['c'])
#c_out = tf.decode_raw(c_raw_out, tf.uint8)
print( a_out)
print( b_out)
print( c_out)

a_batch, b_batch, c_batch = tf.train.shuffle_batch([a_out, b_out, c_out], batch_size=10,
                                capacity=200, min_after_dequeue=100, num_threads=2)


print( a_batch)
print( b_batch)
print( c_batch)