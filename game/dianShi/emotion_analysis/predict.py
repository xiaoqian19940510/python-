import tensorflow as tf
# sess = tf.Session()

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import data_helpers
import word2vec_helpers
import os

tf.flags.DEFINE_string("test_file", "./data/data_test.csv", "Data source for the mid data.")
FLAGS = tf.flags.FLAGS

print("Loading data...")
x_text, y = data_helpers.load_test_files(FLAGS.test_file)
sentences, max_document_length = data_helpers.padding_sentences(x_text, '<PADDING>')
x = np.array(word2vec_helpers.embedding_sentences(sentences, embedding_size = 128, file_to_save = os.path.join('data/', 'trained_word2vec.model')))
print("x.shape = {}".format(x.shape))
print("y.shape = {}".format(y.shape))
# data=pd.read_csv('data/data_test.csv',encoding='utf-8')
# test=pd.DataFrame(data)
# W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weights")
# b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")
# saver = tf.train.Saver()


with tf.Session() as sess:
    saver=tf.train.import_meta_graph('/home/liqian/liqian/NLP/runs/1525330118/checkpoints/model-1100.meta')
    saver.restore(sess,tf.train.latest_checkpoint("/home/liqian/liqian/NLP/runs/1525330118/checkpoints/"))
    graph = tf.get_default_graph()
    input_x=graph.get_operation_by_name('input_x').outputs[0]
    y=tf.get_collection("scores")
    # y = graph.get_operation_by_name('predictions').outputs[0]
    # keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]
    keep_prob = tf.get_collection("dropout_keep_prob")
    pre_y = sess.run(y, feed_dict={input_x: x, keep_prob: 1.0})
    # input_x = graph.get_tensor_by_name("input_x:0")
    # feed_dict = {input_x: data}
    # W = graph.get_tensor_by_name("W:0")
    # h_drop = graph.get_tensor_by_name("dropout:0")
    # b = tf.Variable(tf.constant(0.1, shape=[3], name="b"))
    # sess.run('b:0')
    # scores = tf.nn.xw_plus_b(feed_dict,W, b, name="scores")
    # predictions = tf.argmax(scores, 1, name="predictions")
    # classification_result = sess.run(logits, feed_dict)
    # output = []
    # output = tf.argmax(classification_result,1).eval()
    pre_y.to_csv('/home/liqian/liqian/NLP/predict.csv')
    # saver.restore(sess, "/home/liqian/liqian/NLP/runs/1525330118/checkpoints/model-1100")

# saver.restore(sess, "/home/liqian/liqian/NLP/runs/1525330118/checkpoints/model-1100.ckpt")

# scores = tf.nn.xw_plus_b(W, b, name = "scores")
# result = sess.run(y, feed_dict={x: test})



# import tensorflow as tf
# with tf.Session() as sess:
#     new_saver=tf.train.import_meta_graph('/home/liqian/liqian/NLP/runs/1525330118/checkpoints/model-1100.meta')
#     new_saver.restore(sess,"/home/liqian/liqian/NLP/runs/1525330118/checkpoints/model-1100")
#     graph = tf.get_default_graph()
#     test=graph.get_operation_by_name('x').outputs[0]
#     y=tf.get_collection("pred_network")[0]
#     print("109的预测值是:",sess.run(y, feed_dict={test: x}))
