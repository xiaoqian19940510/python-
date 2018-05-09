import numpy as np
from sklearn import preprocessing
import tensorflow as tf
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
# 波士顿房价数据
# boston = load_boston()
# x = boston.data
# y = boston.target
# x_3 = x[:, 3:6]
# x = np.column_stack([x, x_3])  # 随意给x增加了3列，x变为16列，可以reshape为4*4矩阵了 没啥用，就是凑个正方形

#Ki predict
train=pd.read_csv('taijing/df_affinity_train_combine.csv')
test=pd.read_csv('taijing/df_affinity_test_combine.csv')
# label=pd.read_csv('taijing/df_affinity_train_label.csv')
train=pd.DataFrame(train)
test=pd.DataFrame(test)
label=train['Ki']
label.columns=['ID','Ki']
# del label['ID']
del train['Ki']
test.columns = ['ID','Protein_ID','Molecule_ID','cyp_3a4','cyp_2c9','cyp_2d6','ames_toxicity','fathead_minnow_toxicity','tetrahymena_pyriformis_toxicity','honey_bee','cell_permeability','logP','renal_organic_cation_transporter','CLtotal','hia','biodegradation','Vdd','p_glycoprotein_inhibition','NOAEL','solubility','bbb']
train.columns = ['ID','Protein_ID','Molecule_ID','cyp_3a4','cyp_2c9','cyp_2d6','ames_toxicity','fathead_minnow_toxicity','tetrahymena_pyriformis_toxicity','honey_bee','cell_permeability','logP','renal_organic_cation_transporter','CLtotal','hia','biodegradation','Vdd','p_glycoprotein_inhibition','NOAEL','solubility','bbb']
train.fillna(0.0,inplace=True)
test.fillna(0.0,inplace=True)

# train = shuffle(train)
# label=train['Ki']
# del train['Ki']
del train['ID']
del test['ID']
train.insert(4,'a',train['cyp_3a4'])
train.insert(4,'b',train['cyp_2d6'])
train.insert(4,'c',train['ames_toxicity'])
train.insert(4,'d',train['fathead_minnow_toxicity'])
train.insert(4,'e',train['honey_bee'])
test.insert(4,'a',test['cyp_3a4'])
test.insert(4,'b',test['cyp_2d6'])
test.insert(4,'c',test['ames_toxicity'])
test.insert(4,'d',test['fathead_minnow_toxicity'])
test.insert(4,'e',test['honey_bee'])
test.fillna(0.0,inplace=True)




print('##################################################################')

# 随机挑选
train_x_disorder, test_x_disorder, train_y_disorder, test_y_disorder = train_test_split(train, label, train_size=0.75, random_state=33)
# 数据标准化
ss_x = preprocessing.StandardScaler()
train_x_disorder = ss_x.fit_transform(train_x_disorder)
test_x_disorder = ss_x.transform(test_x_disorder)
pre_x_disorder = ss_x.transform(test)

ss_y = preprocessing.StandardScaler()
train_y_disorder = ss_y.fit_transform(train_y_disorder.values.reshape(-1, 1))
test_y_disorder = ss_y.transform(test_y_disorder.values.reshape(-1, 1))
# train_y_disorder = ss_y.fit_transform(train_y_disorder)
# test_y_disorder = ss_y.transform(test_y_disorder)

# 准确率计算
# def compute_accuracy(v_xs, v_ys):
#     global prediction
#     y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
#     correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
#     return result

# 变厚矩阵
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 偏置
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积处理 变厚过程
def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1] x_movement、y_movement就是步长
    # Must have strides[0] = strides[3] = 1 padding='SAME'表示卷积后长宽不变
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# pool 长宽缩小一倍
def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 25],name='x')  # 原始数据的维度：16
ys = tf.placeholder(tf.float32, [None, 1])  # 输出数据为维度：1
# pre_x= tf.reshape(pre_x_disorder, [-1, 5, 5, 1])
keep_prob = tf.placeholder(tf.float32)  # dropout的比例

x_image = tf.reshape(xs, [-1, 5, 5, 1])  # 原始数据16变成二维图片4*4
## conv1 layer ##第一卷积层
W_conv1 = weight_variable([2, 2, 1, 32])  # patch 2x2, in size 1, out size 32,每个像素变成32个像素，就是变厚的过程
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 2x2x32，长宽不变，高度为32的三维图像
# h_pool1 = max_pool_2x2(h_conv1)     # output size 2x2x32 长宽缩小一倍

## conv2 layer ##第二卷积层
W_conv2 = weight_variable([2, 2, 32, 64])  # patch 2x2, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)  # 输入第一层的处理结果 输出shape 4*4*64

## conv2 layer ##第二卷积层
W_conv3 = weight_variable([2, 2, 64, 128])  # patch 2x2, in size 32, out size 64
b_conv3 = bias_variable([128])
h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)  # 输入第一层的处理结果 输出shape 4*4*64

## conv2 layer ##第二卷积层
W_conv4 = weight_variable([2, 2, 128, 256])  # patch 2x2, in size 32, out size 64
b_conv4 = bias_variable([256])
h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)  # 输入第一层的处理结果 输出shape 4*4*64
## conv2 layer ##第二卷积层
W_conv5 = weight_variable([2, 2, 256, 128])  # patch 2x2, in size 32, out size 64
b_conv5 = bias_variable([128])
h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)  # 输入第一层的处理结果 输出shape 4*4*64

## conv2 layer ##第二卷积层
W_conv6 = weight_variable([2, 2, 128, 256])  # patch 2x2, in size 32, out size 64
b_conv6 = bias_variable([256])
h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6) + b_conv6)  # 输入第一层的处理结果 输出shape 4*4*64
## conv2 layer ##第二卷积层
W_conv7 = weight_variable([2, 2, 256, 128])  # patch 2x2, in size 32, out size 64
b_conv7 = bias_variable([128])
h_conv7 = tf.nn.relu(conv2d(h_conv6, W_conv7) + b_conv7)  # 输入第一层的处理结果 输出shape 4*4*64

## conv2 layer ##第二卷积层
W_conv8 = weight_variable([2, 2, 128, 256])  # patch 2x2, in size 32, out size 64
b_conv8 = bias_variable([256])
h_conv8 = tf.nn.relu(conv2d(h_conv7, W_conv8) + b_conv8)  # 输入第一层的处理结果 输出shape 4*4*64

## conv2 layer ##第二卷积层
W_conv9 = weight_variable([2, 2, 256, 512])  # patch 2x2, in size 32, out size 64
b_conv9 = bias_variable([512])
h_conv9 = tf.nn.relu(conv2d(h_conv8, W_conv9) + b_conv9)  # 输入第一层的处理结果 输出shape 4*4*64
## conv2 layer ##第二卷积层
W_conv10 = weight_variable([2, 2, 512, 1024])  # patch 2x2, in size 32, out size 64
b_conv10= bias_variable([1024])
h_conv10= tf.nn.relu(conv2d(h_conv9, W_conv10) + b_conv10)  # 输入第一层的处理结果 输出shape 4*4*64

## conv2 layer ##第二卷积层
W_conv11= weight_variable([2, 2, 1024, 512])  # patch 2x2, in size 32, out size 64
b_conv11= bias_variable([512])
h_conv11= tf.nn.relu(conv2d(h_conv10, W_conv11) + b_conv11)  # 输入第一层的处理结果 输出shape 4*4*64
## conv2 layer ##第二卷积层
W_conv12= weight_variable([2, 2, 512, 256])  # patch 2x2, in size 32, out size 64
b_conv12= bias_variable([256])
h_conv12= tf.nn.relu(conv2d(h_conv11, W_conv12) + b_conv12)  # 输入第一层的处理结果 输出shape 4*4*64
## conv2 layer ##第二卷积层
W_conv13= weight_variable([2, 2, 256, 128])  # patch 2x2, in size 32, out size 64
b_conv13= bias_variable([128])
h_conv13= tf.nn.relu(conv2d(h_conv12, W_conv13) + b_conv13)  # 输入第一层的处理结果 输出shape 4*4*64
## conv2 layer ##第二卷积层
W_conv14= weight_variable([2, 2, 128, 64])  # patch 2x2, in size 32, out size 64
b_conv14= bias_variable([64])
h_conv14= tf.nn.relu(conv2d(h_conv13, W_conv14) + b_conv14)  # 输入第一层的处理结果 输出shape 4*4*64
## fc1 layer ##  full connection 全连接层
W_fc1 = weight_variable([5 * 5 * 64, 1024])  # 4x4 ，高度为64的三维图片，然后把它拉成512长的一维数组
b_fc1 = bias_variable([1024])

h_pool14_flat = tf.reshape(h_conv14, [-1, 5 * 5 * 64])  # 把4*4，高度为64的三维图片拉成一维数组 降维处理
h_fc1 = tf.nn.relu(tf.matmul(h_pool14_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # 把数组中扔掉比例为keep_prob的元素
## fc2 layer ## full connection
W_fc2 = weight_variable([1024, 1])  # 512长的一维数组压缩为长度为1的数组
b_fc2 = bias_variable([1])  # 偏置
# 最后的计算结果
prediction = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
tf.add_to_collection('pred_network', prediction)
# prediction = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
# 计算 predition与y 差距 所用方法很简单就是用 suare()平方,sum()求和,mean()平均值
cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
# 0.01学习效率,minimize(loss)减小loss误差
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
sess.run(tf.global_variables_initializer())
# 训练500次
saver = tf.train.Saver()
for i in range(50000):
    sess.run(train_step, feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 0.75})
    print(i, '误差=',
          sess.run(cross_entropy, feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 1.0}))  # 输出loss值

saver_path = saver.save(sess, "taijing/model.ckpt",global_step=100)
print("model saved in file: ", saver_path)
with tf.Session() as sess:
    new_saver=tf.train.import_meta_graph('taijing/model.ckpt-100.meta')
    new_saver.restore(sess,"taijing/model.ckpt-100")
    graph = tf.get_default_graph()
    x=graph.get_operation_by_name('x').outputs[0]
    y=tf.get_collection("pred_network")[0]
    pre_y=sess.run(y, feed_dict={x: pre_x_disorder, keep_prob: 1.0})
    pre_y=pd.DataFrame(pre_y)
    pre_y.insert(0,'id',range(1,41383+1))
    pre_y.to_csv('taijing/result_traijing4.csv')

# 可视化
# prediction_value = sess.run(prediction, feed_dict={xs: test_x_disorder, ys: test_y_disorder, keep_prob: 1.0})
# prediction_value = sess.run(prediction, feed_dict={xs: test_x_disorder, ys: test_y_disorder, keep_prob: 1.0})


###画图###########################################################################
# import matplotlib.pyplot as plt
#
# fig = plt.figure(figsize=(20, 3))  # dpi参数指定绘图对象的分辨率，即每英寸多少个像素，缺省值为80
# axes = fig.add_subplot(1, 1, 1)
# line1, = axes.plot(range(len(prediction_value)), prediction_value, 'b--', label='cnn', linewidth=2)
# # line2,=axes.plot(range(len(gbr_pridict)), gbr_pridict, 'r--',label='优选参数')
# line3, = axes.plot(range(len(test_y_disorder)), test_y_disorder, 'g', label='实际')
#
# axes.grid()
# fig.tight_layout()
# # plt.legend(handles=[line1, line2,line3])
# plt.legend(handles=[line1, line3])
# plt.title('卷积神经网络')
# plt.show()


