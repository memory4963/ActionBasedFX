import tensorflow as tf
# import numpy as np

if __name__ == '__main__':
    with tf.Session as sess:
        saver = tf.train.import_meta_graph('D:\\ActionBasedFX\\origin_output\\model_950.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('D:\\ActionBasedFX\\origin_output\\'))

        graph = tf.get_default_graph()
        x0 = graph.get_tensor_by_name('x0:0')
        x1 = graph.get_tensor_by_name('x1:0')
        x5 = graph.get_tensor_by_name('x5:0')
        x10 = graph.get_tensor_by_name('x10:0')
        batch_size = graph.get_tensor_by_name('batch_size:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        y = graph.get_tensor_by_name('y:0')

        tf.saved_model.simple_save(sess, 'D:\\ActionBasedFX\\origin_output\\saved_model',
                                   inputs={'x0': x0, 'x1': x1, 'x5': x5, 'x10': x10,
                                           'batch_size': batch_size, 'keep_prob': keep_prob},
                                   outputs={'y': y})
