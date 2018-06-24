import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
#
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# print(type(mnist.train.images))
# s = "1.1 2.2 3.3 " \
#     "4.4 5.5 6.6"
# print(np.fromstring(s, dtype=np.float32, sep=" "))
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.tostring('C'))
arr2 = np.fromstring(arr.tostring())
