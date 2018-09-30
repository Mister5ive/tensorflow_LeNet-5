# https://blog.csdn.net/enchanted_zhouh/article/details/76855108
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from PIL import Image

WIDTH = 32
HEIGHT = 32

training_epochs = 20000
training_batch = 32
display_step = 50
test_step = 100

mnist = input_data.read_data_sets('../MNIST_data/',one_hot=True)
x_test = mnist.test.images
y_test = mnist.test.labels

xs = tf.placeholder(tf.float32,[None,WIDTH,HEIGHT],name='x_data')
ys = tf.placeholder(tf.float32,[None,10],name='y_data')
keep_prob = tf.placeholder(tf.float32,name='keep_prob')

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

# x:输入 W：权重
def conv2d(x, W, padding):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)

def relu_bias(x,bias,name):
    return tf.nn.relu(tf.nn.bias_add(x,bias),name=name)

# x:输入
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def inference(_input,_reg):

    x_data = tf.reshape(_input,[-1,32,32,1])
    # 第一层：卷积层，过滤器的尺寸为5×5，深度为6,不使用全0补充，步长为1。
    # 尺寸变化：32×32×1->28×28×6
    with tf.variable_scope('Layer1_Conv'):
        conv1_weight = weight_variable([5,5,1,6],'conv1_weight')
        tf.summary.histogram('Layer1_Conv/weights',conv1_weight)
        conv1_bias = bias_variable([6],'conv1_bias')
        tf.summary.histogram('Layer1_Conv/biases', conv1_bias)
        conv1 = conv2d(x_data,conv1_weight,'VALID')
        conv1_relu = relu_bias(conv1,conv1_bias,'conv1_relu')
        tf.summary.histogram('Layer1_Conv/output', conv1_relu)

    # 第二层：池化层，过滤器的尺寸为2×2，使用全0补充，步长为2。
    # 尺寸变化：28×28×6->14×14×6
    with tf.name_scope('Layer2_Pooling'):
        pool2 = max_pool_2x2(conv1_relu)
        tf.summary.histogram('Layer2_Pooling/output', pool2)

    # 第三层：卷积层，过滤器的尺寸为5×5，深度为16,不使用全0补充，步长为1。
    # 尺寸变化：14×14×6->10×10×16
    with tf.variable_scope('Layer3_Conv'):
        conv3_weight = weight_variable([5, 5, 6, 16], 'conv3_weight')
        tf.summary.histogram('Layer3_Conv/weights', conv3_weight)
        conv3_bias = bias_variable([16], 'conv2_bias')
        tf.summary.histogram('Layer3_Conv/biases', conv3_bias)
        conv3 = conv2d(pool2, conv3_weight, 'VALID')
        conv3_relu = relu_bias(conv3, conv3_bias, 'conv3_relu')
        tf.summary.histogram('Layer3_Conv/output', conv3_relu)

    # 第四层：池化层，过滤器的尺寸为2×2，使用全0补充，步长为2。
    # 尺寸变化：10×10×6->5×5×16
    with tf.variable_scope('Layer4_Pooling'):
        pool4 = max_pool_2x2(conv3_relu)
        tf.summary.histogram('Layer4_Pooling/output', pool4)

    # 原文第五层是5*5的卷积层，因为输入是5*5*16的map，所以这里即相当于一个全连接层。
    # 5×5×16->1 x 400
    pool4_shape = pool4.get_shape().as_list()
    size = pool4_shape[1] * pool4_shape[2] * pool4_shape[3]
    pool4_reshape = tf.reshape(pool4,[-1,size])


    # 第五层：全连接层，nodes=5×5×16=400，400->120的全连接
    # 尺寸变化：比如一组训练样本为64，那么尺寸变化为64×400->64×120
    # 训练时，引入dropout，dropout在训练时会随机将部分节点的输出改为0，dropout可以避免过拟合问题。
    # 这和模型越简单越不容易过拟合思想一致，和正则化限制权重的大小，使得模型不能任意拟合训练数据中的随机噪声，以此达到避免过拟合思想一致。
    with tf.variable_scope('Layer5_FC'):

        fc5_weight = weight_variable([size,120],'fc5_weight')
        tf.summary.histogram('Layer5_FC/weights', fc5_weight)
        fc5_bias = bias_variable([120],'fc5_bias')
        tf.summary.histogram('Layer5_FC/biases', fc5_bias)
        if _reg != None:
            tf.add_to_collection('losses', _reg(fc5_weight))
        fc5 = tf.matmul(pool4_reshape,fc5_weight)
        fc5_relu = relu_bias(fc5,fc5_bias,'fc5_relu')
        fc5_relu = tf.nn.dropout(fc5_relu, keep_prob)
        tf.summary.histogram('Layer5_FC/output', fc5_relu)

    # 第六层：全连接层，120->84的全连接
    # 尺寸变化：比如一组训练样本为64，那么尺寸变化为64×120->64×84
    with tf.variable_scope('Layer6_FC'):

        fc6_weight = weight_variable([120, 84], 'fc6_weight')
        tf.summary.histogram('Layer6_FC/weights', fc6_weight)
        fc6_bias = bias_variable([84], 'fc6_bias')
        tf.summary.histogram('Layer6_FC/biases', fc6_bias)
        if _reg != None:
            tf.add_to_collection('losses', _reg(fc6_weight))
        fc6 = tf.matmul(fc5_relu, fc6_weight)
        fc6_relu = relu_bias(fc6, fc6_bias,'fc6_relu')
        fc6_relu = tf.nn.dropout(fc6_relu, keep_prob)
        tf.summary.histogram('Layer6_FC/output', fc6_relu)

    # 第七层：全连接层（近似表示），84->10的全连接
    # 尺寸变化：比如一组训练样本为64，那么尺寸变化为64×84->64×10。最后，64×10的矩阵经过softmax之后就得出了64张图片分类于每种数字的概率，
    # 即得到最后的分类结果。
    with tf.variable_scope('Layer7_FC'):

        fc7_weight = weight_variable([84, 10], 'fc7_weight')
        tf.summary.histogram('Layer7_FC/weights', fc7_weight)
        fc7_bias = bias_variable([10], 'fc7_bias')
        tf.summary.histogram('Layer7_FC/biases', fc7_bias)
        if _reg != None:
            tf.add_to_collection('losses', _reg(fc7_weight))
        result = tf.matmul(fc6_relu, fc7_weight) + fc7_bias
        tf.summary.histogram('Layer7_FC/output', result)
    return result

# padding mnist 28x28->32x32
def mnist_reshape_32(_batch):

    batch = np.reshape(_batch,[-1,28,28])
    num = batch.shape[0]
    batch_32 = np.array(np.random.rand(num,32,32),dtype=np.float32)
    for i in range(num):
        batch_32[i] = np.pad(batch[i],2,'constant',constant_values=0)

    return batch_32



regularizer = tf.contrib.layers.l2_regularizer(0.001)
y = inference(xs,regularizer)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(ys,1))
with tf.name_scope('loss'):
    loss = tf.reduce_mean(cross_entropy) + tf.add_n(tf.get_collection('losses'))
    tf.summary.scalar('loss',loss)

with tf.name_scope('loss'):
    train_step = tf.train.GradientDescentOptimizer(.01).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(ys,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

x_test_32 = mnist_reshape_32(x_test)


with tf.Session() as sess:

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('logs/', sess.graph)

    sess.run(init)
    for i in range(training_epochs):
        batch_xs, batch_ys = mnist.train.next_batch(training_batch) # 28x28
        batch_xs_32 = mnist_reshape_32(batch_xs)

        sess.run(train_step,feed_dict={xs:batch_xs_32,ys:batch_ys,keep_prob:.3})

        if i % display_step == 0:
            print('---------------step:%d, training accuracy:%g--------------' % (i, sess.run(accuracy,feed_dict={
                xs: batch_xs_32, ys: batch_ys, keep_prob: 1.0})))
            rs = sess.run(merged,feed_dict={xs: batch_xs_32, ys: batch_ys, keep_prob: 1.0})
            writer.add_summary(rs,i)

        if i % test_step == 0:
            print("test accuracy %g" % sess.run(accuracy, feed_dict={
                xs: x_test_32, ys: y_test, keep_prob: 1.0}))

    print('---------------step:%d, training accuracy:%g--------------' % (i, sess.run(accuracy, feed_dict={
        xs: batch_xs_32, ys: batch_ys, keep_prob: 1.0})))