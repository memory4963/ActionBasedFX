import random
import tensorflow as tf
from tensorflow.contrib import rnn
import ReadData
import Utils


temp = Utils.arg_proc()

start_gpu = temp.start_gpu
gpu_num = temp.gpu_num
_batch_size = temp.batch_size
dataset_size = temp.dataset_size  # zero means no limitation
output_path = temp.output_path

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

skeleton, labels = ReadData.read_data("/home/luoao/openpose/dataset/simpleOutput", dataset_size)

# learning rate
lr = 1e-4
batch_size = tf.placeholder(tf.int32, [])
# 36 per frame
input_size = 36
# 60 frame per batch
timestep_size = ReadData.data_length

hidden_size = 1024
# LSTM layer num
layer_num = 2
# output num
class_num = labels.shape[1]


def make_lstm():
    lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
    return rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)


x = tf.placeholder(tf.float32, [None, timestep_size, input_size])
label = tf.placeholder(tf.float32, [None, class_num])
keep_prob = tf.placeholder(tf.float32, [])

with tf.name_scope("lstm_cell"):
    mlstm_cell = rnn.MultiRNNCell([make_lstm() for _ in range(layer_num)], state_is_tuple=True)
init_state = mlstm_cell.zero_state(batch_size, tf.float32)
# lstm_cell * layer_num
outputs, _ = tf.nn.dynamic_rnn(mlstm_cell, inputs=x, initial_state=init_state, time_major=False)
with tf.name_scope("hidden_state"):
    h_state = outputs[:, -1, :]
# hidden layer * 2
with tf.name_scope("full_weights1"):
    full_weights1 = tf.Variable(tf.truncated_normal([hidden_size, hidden_size / 4], stddev=1e-2), dtype=tf.float32)
    Utils.variable_summaries(full_weights1)
with tf.name_scope("full_bias1"):
    full_bias1 = tf.Variable(tf.constant(1e-2, shape=[hidden_size / 4]))
    Utils.variable_summaries(full_bias1)
hidden = tf.nn.relu(tf.add(tf.matmul(h_state, full_weights1), full_bias1))
with tf.name_scope("full_weights2"):
    full_weights2 = tf.Variable(tf.truncated_normal([hidden_size / 4, hidden_size / 16], stddev=1e-2), dtype=tf.float32)
    Utils.variable_summaries(full_weights2)
with tf.name_scope("full_bias2"):
    full_bias2 = tf.Variable(tf.constant(1e-2, shape=[hidden_size / 16]))
    Utils.variable_summaries(full_bias2)
hidden = tf.nn.relu(tf.add(tf.matmul(hidden, full_weights2), full_bias2))
# softmax
with tf.name_scope("weights"):
    weights = tf.Variable(tf.truncated_normal([hidden_size / 16, class_num], stddev=1e-2), dtype=tf.float32)
    Utils.variable_summaries(weights)
with tf.name_scope("bias"):
    bias = tf.Variable(tf.constant(1e-2, shape=[class_num]), dtype=tf.float32)
    Utils.variable_summaries(bias)
y = tf.nn.softmax(tf.add(tf.matmul(hidden, weights), bias))

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
train_writer = tf.summary.FileWriter(output_path + "/train", sess.graph)
test_writer = tf.summary.FileWriter(output_path + "/test")

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

for i in range(1000):
    if (i + 1) % 20 == 0:
        # test accuracy
        j = int(float(i) / 1000 * skeleton.shape[0] / _batch_size - 1)
        test_batch = skeleton[j * _batch_size:j * _batch_size + _batch_size, :, :]
        test_labels = labels[j * _batch_size:j * _batch_size + _batch_size]
        summary, train_accuracy, loss = sess.run([merged, accuracy, cross_entropy], feed_dict={
            x: test_batch, label: test_labels,
            keep_prob: 1.0, batch_size: _batch_size
        })
        test_writer.add_summary(summary, i * skeleton.shape[0] / _batch_size)
        print("train step %d, accuracy: %f, loss:%f" % (i, train_accuracy, loss))
    if (i + 1) % 100 == 0:
        # save
        saver.save(sess, output_path + "/model_" + str(i) + ".ckpt")

    train_test_int = random.randint(0, skeleton.shape[0] / _batch_size)
    for j in range(skeleton.shape[0] / _batch_size):
        train_batch = skeleton[j * _batch_size:j * _batch_size + _batch_size, :, :]
        train_labels = labels[j * _batch_size:j * _batch_size + _batch_size]
        summary, _ = sess.run([merged, optimizer], feed_dict={
            x: train_batch, label: train_labels,
            keep_prob: 0.8, batch_size: _batch_size
        })
        if j == train_test_int:
            train_writer.add_summary(summary, i * skeleton.shape[0] / _batch_size + j)
