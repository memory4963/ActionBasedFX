import tensorflow as tf
# import numpy as np

if __name__ == '__main__':
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('D:\\ActionBasedFX\\origin_output\\model_950.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('D:\\ActionBasedFX\\origin_output\\'))

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name('x:0')
        batch_size = graph.get_tensor_by_name('batch_size:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        y = graph.get_tensor_by_name('y:0')

        builder = tf.saved_model.builder.SavedModelBuilder('D:\\ActionBasedFX\\origin_output\\saved_model')
        input_dict = {'x': tf.saved_model.utils.build_tensor_info(x),
                      'batch_size': tf.saved_model.utils.build_tensor_info(batch_size),
                      'keep_prob': tf.saved_model.utils.build_tensor_info(keep_prob)}
        output_dict = {'y': tf.saved_model.utils.build_tensor_info(y)}
        signature = tf.saved_model.signature_def_utils.build_signature_def(input_dict, output_dict, 'ts_predict')
        builder.add_meta_graph_and_variables(sess, ['TS_LSTM'], {'predict_sig': signature})
        builder.save()
