import tensorflow as tf
from tensorflow.contrib import rnn
import ReadData
import Utils

lr = 1e-4
input_size = 36
timestep_size = ReadData.data_length
batch_size = tf.placeholder(tf.int32, [])

hidden_size = 256
layer_num = 2
keep_prob = tf.placeholder(tf.float32, [])

# TS-LSTM networks. set time step to 36
# 0: hidden 128 D 1  W 8  TS 8
# 1: hidden 64  D 1  W 20 TS 15
# 2: hidden 64  D 5  W 16 TS 15
# 3: hidden 32  D 1  W 35 TS -
# 4: hidden 32  D 5  W 32 TS -
# 5: hidden 32  D 10 W 28 TS -
# 6: hidden 64  D 0  W 16 TS 8
# input0->(TS-LSTM0)->(SumPool)->(Linear)->(Dropout)->(SoftMax)->Softmax0
#
# input1->(TS-LSTM1)->(SumPool)-\
#                               ->(Concat)->(Linear)->(Dropout)->(SoftMax)->Softmax1
# input2->(TS-LSTM2)->(SumPool)-/
#
# input3->(TS-LSTM3)->(SumPool)-\
# input4->(TS-LSTM4)->(SumPool)--->(Concat)->(Linear)->(Dropout)->(SoftMax)->Softmax2
# input5->(TS-LSTM5)->(SumPool)-/
#
# input6->(TS-LSTM6)->(MeanPool)->(Linear)->(Dropout)->(SoftMax)->Softmax
#
# Softmax0--\
# Softmax1-\ \
#           --->(Concat)->output/class
# Softmax2-/ /
# Softmax3--/
windows = [(timestep_size - 1) / 4,
           (timestep_size - 1) / 2,
           (timestep_size - 5) / 2,
           timestep_size - 1,
           timestep_size - 5,
           timestep_size - 10,
           timestep_size / 2]

ts = windows

linear_size = [128, 64, 32, 64]


# lstm_cnt = 0


def lstm_cell(shape):
    # global lstm_cnt
    # with tf.variable_scope('lstm' + str(lstm_cnt)):
    return rnn.BasicLSTMCell(shape, state_is_tuple=True, forget_bias=1.0, reuse=tf.AUTO_REUSE)


def ts_lstm(window):
    return [lstm_cell(hidden_size) for _ in range(window)]


if __name__ == '__main__':
    # process args
    args = Utils.arg_proc()

    # tf.reset_default_graph()

    # load data set
    skeleton, labels = ReadData.read_data("/home/luoao/openpose/dataset/simpleOutput", args.dataset_size)
    skeleton1 = skeleton[:, 1:, :] - skeleton[:, :-1, :]
    skeleton5 = skeleton[:, 5:, :] - skeleton[:, :-5, :]
    skeleton10 = skeleton[:, 10:, :] - skeleton[:, :-10, :]
    class_num = labels.shape[1]

    # declare placeholders
    x0 = tf.placeholder(tf.float32, [None, timestep_size, input_size], name="x0")
    x1 = tf.placeholder(tf.float32, [None, timestep_size - 1, input_size], name="x1")
    x5 = tf.placeholder(tf.float32, [None, timestep_size - 5, input_size], name="x5")
    x10 = tf.placeholder(tf.float32, [None, timestep_size - 10, input_size], name="x10")
    label = tf.placeholder(tf.float32, [None, class_num])

    x = [x1, x1, x5, x1, x5, x10, x0]

    # network structure
    ts_lstms = [ts_lstm(windows[0]),
                ts_lstm(windows[1]),
                ts_lstm(windows[2]),
                ts_lstm(windows[3]),
                ts_lstm(windows[4]),
                ts_lstm(windows[5]),
                ts_lstm(windows[6])]

    # TS-LSTM [7, window, batch_size, timestep_size, data_length]
    initial_states = [[lstm.zero_state(batch_size, tf.float32) for lstm in tslstm] for tslstm in ts_lstms]
    ts_lstms_outputs = []
    for i, tslstm in enumerate(ts_lstms):
        ts_lstm_output = []
        for j, lstm in enumerate(tslstm):
            lstm_output, _ = tf.nn.dynamic_rnn(lstm,
                                               x[i][:, j::ts[i], :],
                                               initial_state=initial_states[i][j])
            ts_lstm_output.append(lstm_output)
        ts_lstms_outputs.append(ts_lstm_output)

    # SumPool and MeanPool [7, batch_size, data_length]
    pools = []
    for i, outputs in enumerate(ts_lstms_outputs[:-1]):
        print(i, outputs[0].shape)
        sumpool = outputs[0][:, -1]
        for j, output in enumerate(outputs[1:]):
            print(i, j, output.shape)
            sumpool += output[:, -1]
        pools.append(sumpool)
    meanpool = ts_lstms_outputs[-1][0][:, -1]
    for output in ts_lstms_outputs[-1][1:]:
        meanpool += output[:, -1]
    meanpool /= windows[6]
    pools.append(meanpool)

    # Concat [4, batch_size, *]
    # * = [data_length, 2*data_length, 3*data_length, data_length]
    concats = [pools[0]]
    concats.append(tf.concat([pools[1], pools[2]], 1))
    concats.append(tf.concat([pools[3], pools[4], pools[5]], 1))
    concats.append(pools[6])

    # Linear [4, batch_size, linear_size]
    weights = [Utils.weight_variable([concats[i].shape[1].value, linear_size[i]]) for i in range(4)]
    bias = [Utils.bias_variable([linear_size[i]]) for i in range(4)]
    linears = [tf.nn.relu(tf.add(tf.matmul(concats[i], weights[i]), bias[i])) for i in range(4)]

    # Dropout [4, batch_size, linear_size]
    dropouts = [tf.nn.dropout(linear, keep_prob) for linear in linears]

    # SoftMax [4, batch_size, linear_size]
    softmaxes = [tf.nn.softmax(dropout) for dropout in dropouts]

    # Concat [batch_size, sum(linear_size)]
    concat = tf.concat([softmax for softmax in softmaxes], 1)

    # Full Connection
    concat_size = sum(linear_size)
    fc_weights = Utils.weight_variable([concat_size, concat_size / 4])
    fc_bias = Utils.bias_variable([concat_size / 4])
    fc = tf.nn.relu(tf.add(tf.matmul(concat, fc_weights), fc_bias))

    fc_weights = Utils.weight_variable([concat_size / 4, concat_size / 16])
    fc_bias = Utils.bias_variable([concat_size / 16])
    fc = tf.nn.relu(tf.add(tf.matmul(fc, fc_weights), fc_bias))

    # Softmax
    sm_weights = Utils.weight_variable([concat_size / 16, class_num])
    sm_bias = Utils.bias_variable([class_num])
    y = tf.nn.softmax(tf.add(tf.matmul(fc, sm_weights), sm_bias))

    # loss
    cross_entropy = -tf.reduce_mean(label * tf.log(y))
    optimizer = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    # train
    for i in range(1000):
        if (i + 1) % 20 == 0:
            # test accuracy
            j = int(float(i) / 1000 * skeleton.shape[0] / args.batch_size - 1)
            test_batch = skeleton[j * args.batch_size:j * args.batch_size + args.batch_size, :, :]
            test_label = labels[j * args.batch_size:j * args.batch_size + args.batch_size]
            train_accuracy, loss = sess.run([accuracy, cross_entropy], feed_dict={
                x0: test_batch,
                x1: test_batch[:, 1:] - test_batch[:, :-1],
                x5: test_batch[:, 5:] - test_batch[:, :-5],
                x10: test_batch[:, 10:] - test_batch[:, :-10],
                label: test_label,
                keep_prob: 1.0,
                batch_size: args.batch_size
            })
            print('train step %d, acc = %f, loss = %f' % (i, train_accuracy, loss))
        if (i + i) % 100 == 0:
            # save
            saver.save(sess, args.output_path + 'model_' + str(i) + 'ckpt')

        for j in range(skeleton.shape[0] / args.batch_size):
            train_batch = skeleton[j * args.batch_size:j * args.batch_size + args.batch_size, :, :]
            train_batch1 = skeleton1[j * args.batch_size:j * args.batch_size + args.batch_size, :, :]
            train_batch5 = skeleton5[j * args.batch_size:j * args.batch_size + args.batch_size, :, :]
            train_batch10 = skeleton10[j * args.batch_size:j * args.batch_size + args.batch_size, :, :]
            train_label = labels[j * args.batch_size:j * args.batch_size + args.batch_size]
            sess.run(optimizer, feed_dict={
                x0: train_batch,
                x1: train_batch1,
                x5: train_batch5,
                x10: train_batch10,
                label: train_label,
                keep_prob: 0.8,
                batch_size: args.batch_size
            })
