import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
import ReadData
import sys
import getopt

start_gpu = 0
gpu_num = 4

if len(sys.argv) > 1:
    # set params
    try:
        opts, args = getopt.getopt(sys.argv[1:], "", ["start_gpu=", "gpu_num="])
    except getopt.GetoptError:
        print("LSTM.py --start_gpu <num> --gpu_num <num>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '--start_gpu':
            start_gpu = int(arg)
        elif opt == '--gpu_num':
            gpu_num = int(arg)
        else:
            print("LSTM.py --start_gpu <num> --gpu_num <num>")
            sys.exit(2)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# print(type(mnist.train.images))

skeleton, labels = ReadData.read_data("/home/luoao/openpose/dataset/simpleOutput")

# 学习速率
lr = 1e-3
batch_size = tf.placeholder(tf.int32, [])
# 每帧36个数
input_size = 36
# 每个batch60帧
timestep_size = ReadData.batch_size
# 每个隐藏层节点数
hidden_size = 256
# LSTM layer数
layer_num = 2
# 输出class数量
class_num = 4


def make_lstm():
    lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
    return rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)


with tf.device('/device:GPU:' + str(start_gpu)):
    x = tf.placeholder(tf.float32, [None, timestep_size, input_size])
    label = tf.placeholder(tf.int64, [None, class_num])
    keep_prob = tf.placeholder(tf.float32, [])

    mlstm_cell = rnn.MultiRNNCell([make_lstm() for _ in range(layer_num)], state_is_tuple=True)
    init_state = mlstm_cell.zero_state(batch_size, tf.float32)

    outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=x, initial_state=init_state, time_major=False)
    h_state = outputs[:, -1, :]

    # softmax
    weights = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32)
    bias = tf.Variable(tf.constant(0.1), dtype=tf.float32)
    y = tf.nn.softmax(tf.matmul(h_state, weights) + bias)

    cross_entropy = -tf.reduce_mean(label * tf.log(y))
    optimizer = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction), "float")

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
for i in range(1000):
    _batch_size = 32
    train_batch = skeleton[i * 32:i * 32 + 32, :, :]
    train_labels = labels[i * 32:i * 32 + 32]
    if i + 1 % 10 == 0:
        # test accuracy
        train_accuracy = sess.run(accuracy, feed_dict={
            x: train_batch, label: train_labels,
            keep_prob: 1.0, batch_size: _batch_size
        })
        print("train step %d, accuracy: %d" % i, train_accuracy)
    if i + 1 % 100 == 0:
        # save
        saver.save(sess, "/home/luoao/openpose/models/model_" + str(i) + ".ckpt")
    sess.run(optimizer, feed_dict={
        x: train_batch, label: train_labels,
        keep_prob: 1.0, batch_size: skeleton.shape[0]
    })
