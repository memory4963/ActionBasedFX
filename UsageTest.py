import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    with tf.Session as sess:
        saver = tf.train.import_meta_graph('/home/luoao/openpose/models/model_999.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('/home/luoao/openpose/models/'))

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name('x:0')
        label = graph.get_tensor_by_name('label:0')
        batch_size = graph.get_tensor_by_name('batch_size:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        y = graph.get_tensor_by_name('y:0')
        feed_dict = {x: np.arange(36*24).reshape(1, 24, 36),
                     label: np.zeros(4),
                     batch_size: 1,
                     keep_prob: 1.0}
        print(sess.run(y, feed_dict=feed_dict))
