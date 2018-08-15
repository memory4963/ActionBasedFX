import os
import sys
import getopt
import tensorflow as tf


class Args:
    start_gpu = 0
    gpu_num = 1
    batch_size = 16
    dataset_size = 0  # zero means no limitation
    output_path = '/home/luoao/openpose/models'


def arg_proc():
    temp = Args()

    if len(sys.argv) > 1:
        # set params
        try:
            opts, args = getopt.getopt(sys.argv[1:], "",
                                       ["start_gpu=", "gpu_num=", "batch_size=", "dataset_size=", "output_path="])
        except getopt.GetoptError:
            print(
                "LSTM.py --start_gpu <num> --gpu_num <num> --batch_size <num> --dataset_size <num> --output_path <path>")
            sys.exit(2)
        for opt, arg in opts:
            if opt == '--start_gpu':
                temp.start_gpu = int(arg)
            elif opt == '--gpu_num':
                temp.gpu_num = int(arg)
            elif opt == '--batch_size':
                temp.batch_size = int(arg)
            elif opt == '--dataset_size':
                temp.dataset_size = int(arg)
            elif opt == '--output_path':
                temp.output_path = arg
            else:
                print("XX.py --start_gpu <num> --gpu_num <num> --batch_size <num> --dataset_size <num> "
                      "--output_path <path>")
                sys.exit(2)

    if not os.path.exists(temp.output_path):
        os.makedirs(temp.output_path)

    visiable_devices = str(temp.start_gpu)
    for i in range(temp.gpu_num - 1):
        visiable_devices += ", " + str(temp.start_gpu + i + 1)

    os.environ["CUDA_VISIBLE_DEVICES"] = visiable_devices
    return temp


def arg_proc_win():
    temp = Args()

    temp.output_path = 'D:\\ts_output'

    if len(sys.argv) > 1:
        # set params
        try:
            opts, args = getopt.getopt(sys.argv[1:], "",
                                       ["start_gpu=", "gpu_num=", "batch_size=", "dataset_size=", "output_path="])
        except getopt.GetoptError:
            print(
                "XX.py --start_gpu <num> --gpu_num <num> --batch_size <num> --dataset_size <num> --output_path <path>")
            sys.exit(2)
        for opt, arg in opts:
            if opt == '--start_gpu':
                temp.start_gpu = int(arg)
            elif opt == '--gpu_num':
                temp.gpu_num = int(arg)
            elif opt == '--batch_size':
                temp.batch_size = int(arg)
            elif opt == '--dataset_size':
                temp.dataset_size = int(arg)
            elif opt == '--output_path':
                temp.output_path = arg
            else:
                print("LSTM.py --start_gpu <num> --gpu_num <num> --batch_size <num> --dataset_size <num> "
                      "--output_path <path>")
                sys.exit(2)

    if not os.path.exists(temp.output_path):
        os.makedirs(temp.output_path)

    visiable_devices = str(temp.start_gpu)
    for i in range(temp.gpu_num - 1):
        visiable_devices += ", " + str(temp.start_gpu + i + 1)

    os.environ["CUDA_VISIBLE_DEVICES"] = visiable_devices
    return temp


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


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
