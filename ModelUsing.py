import tensorflow as tf
import sys
import os
import ReadData
from tensorflow.contrib import rnn
import getopt
import numpy as np

start_gpu = 0
gpu_num = 1
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
        else:
            print("LSTM.py --start_gpu <num> --gpu_num <num>")
            sys.exit(2)

visiable_devices = str(start_gpu)
for i in range(gpu_num - 1):
    visiable_devices += ", " + str(start_gpu + i + 1)

os.environ["CUDA_VISIBLE_DEVICES"] = visiable_devices

print('loading')
sess = tf.Session()

timestep_size = ReadData.data_length
input_size = 36
hidden_size = 1024
class_num = 4
layer_num = 2


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
keep_prob = tf.placeholder(tf.float32, [])
batch_size = tf.placeholder(tf.int32, [])

with tf.name_scope("lstm_cell"):
    mlstm_cell = rnn.MultiRNNCell([make_lstm() for _ in range(layer_num)], state_is_tuple=True)
init_state = mlstm_cell.zero_state(batch_size, tf.float32)
# lstm_cell * layer_num
outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=x, initial_state=init_state, time_major=False)
with tf.name_scope("hidden_state"):
    h_state = outputs[:, -1, :]
# hidden layer * 2
with tf.name_scope("full_weights1"):
    full_weights1 = tf.Variable(tf.truncated_normal([hidden_size, hidden_size / 4], stddev=1e-2), dtype=tf.float32)
    variable_summaries(full_weights1)
with tf.name_scope("full_bias1"):
    full_bias1 = tf.Variable(tf.constant(1e-2, shape=[hidden_size / 4]))
    variable_summaries(full_bias1)
hidden = tf.nn.relu(tf.add(tf.matmul(h_state, full_weights1), full_bias1))
with tf.name_scope("full_weights2"):
    full_weights2 = tf.Variable(tf.truncated_normal([hidden_size / 4, hidden_size / 16], stddev=1e-2), dtype=tf.float32)
    variable_summaries(full_weights2)
with tf.name_scope("full_bias2"):
    full_bias2 = tf.Variable(tf.constant(1e-2, shape=[hidden_size / 16]))
    variable_summaries(full_bias2)
hidden = tf.nn.relu(tf.add(tf.matmul(hidden, full_weights2), full_bias2))
# softmax
with tf.name_scope("weights"):
    weights = tf.Variable(tf.truncated_normal([hidden_size / 16, class_num], stddev=1e-2), dtype=tf.float32)
    variable_summaries(weights)
with tf.name_scope("bias"):
    bias = tf.Variable(tf.constant(1e-2, shape=[class_num]), dtype=tf.float32)
    variable_summaries(bias)
y = tf.nn.softmax(tf.add(tf.matmul(hidden, weights), bias))

saver = tf.train.Saver()
saver.restore(sess, "/home/luoao/openpose/models/model_999.ckpt")


def process_data(inputs, labels):
    this_outputs = sess.run(y, feed_dict={x: inputs, batch_size: inputs.shape[0], keep_prob: 1.0})
    this_outputs = np.argmax(this_outputs, axis=1)
    for i in range(this_outputs.shape[0]):
        if this_outputs[i] == 0:
            this_type = 'marking time and knee lifting. a1'
        elif this_outputs[i] == 1:
            this_type = 'squatting. a3'
        elif this_outputs[i] == 2:
            this_type = 'rotation clapping. a9'
        else:
            this_type = 'punching. a12'
        print('name: ' + labels[i] + ', type: ' + this_type + '\n')


print("please input path of file. input 'exit' to exit\n")
path = sys.stdin.readline()
while path != "exit\n":
    print('path: ' + path)
    path = path[:-1]
    if not os.path.exists(path):
        print('path not exit, please input again.\n')
    else:
        skeleton, names = ReadData.read_data_test(path)
        process_data(skeleton, names)
    path = sys.stdin.readline()
print("exit.")
