import os
from random import random

os.environ['LD_LIBRARY_PATH'] = ':/usr/local/cuda/lib64'


import tensorflow as tf
from tensorflow.contrib import rnn
import ReadData
import sys
import getopt


start_gpu = 0
gpu_num = 1
_batch_size = 32
dataset_size = 0 # zero means no limitation

if len(sys.argv) > 1:
    # set params
    try:
        opts, args = getopt.getopt(sys.argv[1:], "", ["start_gpu=", "gpu_num=", "batch_size=", "dataset_size="])
    except getopt.GetoptError:
        print("LSTM.py --start_gpu <num> --gpu_num <num>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '--start_gpu':
            start_gpu = int(arg)
        elif opt == '--gpu_num':
            gpu_num = int(arg)
        elif opt == '--batch_size':
            _batch_size = int(arg)
        elif opt == '--dataset_size':
            dataset_size = int(arg)
        else:
            print("LSTM.py --start_gpu <num> --gpu_num <num> --batch_size <num> --dataset_size <num>")
            sys.exit(2)

visiable_devices = str(start_gpu)
for i in range(gpu_num - 1):
    visiable_devices += ", " + str(start_gpu + i + 1)

os.environ["CUDA_VISIBLE_DEVICES"] = visiable_devices

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# print(type(mnist.train.images))

skeleton, labels = ReadData.read_data("/home/luoao/openpose/dataset/simpleOutput", dataset_size)

# learning rate
lr = 1e-2
batch_size = tf.placeholder(tf.int32, [])
# 36 per frame
input_size = 36
# 60 frame per batch
timestep_size = ReadData.batch_size

hidden_size = 256
# LSTM layer num
layer_num = 2
# output num
class_num = labels.shape[1]


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def make_lstm():
    lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
    return rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)


x = tf.placeholder(tf.float32, [None, timestep_size, input_size])
label = tf.placeholder(tf.float32, [None, class_num])
keep_prob = tf.placeholder(tf.float32, [])

with tf.name_scope("lstm_cell"):
    mlstm_cell = rnn.MultiRNNCell([make_lstm() for _ in range(layer_num)], state_is_tuple=True)
init_state = mlstm_cell.zero_state(batch_size, tf.float32)

outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=x, initial_state=init_state, time_major=False)
with tf.name_scope("hidden_state"):
    h_state = outputs[:, -1, :]

# softmax
with tf.name_scope("weights"):
    weights = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32)
    variable_summaries(weights)
with tf.name_scope("bias"):
    bias = tf.Variable(tf.constant(0.1, shape=[_batch_size, class_num]), dtype=tf.float32)
    variable_summaries(bias)
y = tf.nn.softmax(tf.matmul(h_state, weights) + bias)

with tf.name_scope("total"):
    cross_entropy = -tf.reduce_mean(label * tf.log(y))
tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(label, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("/home/luoao/openpose/models/train", sess.graph)
test_writer = tf.summary.FileWriter("/home/luoao/openpose/models/test")

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

for i in range(1000):
    if (i + 1) % 20 == 0:
        # test accuracy
        j = int(float(i) / 1000 * skeleton.shape[0] / _batch_size + 0.5)
        test_batch = skeleton[j * _batch_size:j * _batch_size + _batch_size, :, :]
        test_labels = labels[j * _batch_size:j * _batch_size + _batch_size]
        summary, train_accuracy = sess.run([merged, accuracy], feed_dict={
            x: test_batch, label: test_labels,
            keep_prob: 1.0, batch_size: _batch_size
        })
        test_writer.add_summary(summary, i * skeleton.shape[0] / _batch_size)
        print("train step %d, accuracy: %d" % (i, train_accuracy))
    if (i + 1) % 100 == 0:
        # save
        saver.save(sess, "/home/luoao/openpose/models/model_" + str(i) + ".ckpt")

    train_test_int = random.randint(0, skeleton.shape[0] / _batch_size)
    for j in range(skeleton.shape[0] / _batch_size):
        train_batch = skeleton[j * _batch_size:j * _batch_size + _batch_size, :, :]
        train_labels = labels[j * _batch_size:j * _batch_size + _batch_size]
        summary, _ = sess.run([merged, optimizer], feed_dict={
            x: train_batch, label: train_labels,
            keep_prob: 0.5, batch_size: _batch_size
        })
        if j == train_test_int:
            train_writer.add_summary(summary, i * skeleton.shape[0] / _batch_size + j)
