# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 13:02:37 2018

@author: lenovo
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.python.framework import ops
import random

#1.得到置换矩阵
N = random.sample(range(1,1000,1),1)#随机置换的次数
row_change = [round(random.uniform(0, 27)) for i in range(N[0])]#随机置换的行
permutation_matrix=np.eye(28)
def swapRows(M, r1, r2):  
    M[r1],M[r2] = M[r2],M[r1]  
    return(M) 
for i in range(0,N[0]-1):
    permutation_matrix=swapRows(permutation_matrix,row_change[i],row_change[i+1])
    
#2.将MNIST数据集进行置换，并得到one_hot编码的数据
data_dir = 'D:/Anaconda/envs/tensorflow/Lib/urllib/mnist'
mnist = read_data_sets(data_dir,one_hot=True) 
#定义转成矩阵函数
def vector2matrix(a):    #将784转换成28*28的矩阵
    b = np.zeros([28, 28]) #定义一个简单的28X28矩阵
    for i in range(0,27):
        for j in range(0,27):
            b[i][j] = a[28*i+j]
    return b
#定义转成向量函数
#将28x28的二维矩阵，装换成1x784的向量
def matrix2vector (b):
    c = np.zeros(784)
    for i in range(0,27):
        for j in range(0,27):
            c[28*i+j]=b[i][j]
    return c  
  
#得到置换后的训练集、验证集和测试集，标签不变
x1,y1 = np.shape(mnist.train.images)
for i in range(0,x1):
    tile = vector2matrix(mnist.train.images[i])
    tile = np.dot(permutation_matrix,tile)
    tile = np.dot(tile,np.transpose(permutation_matrix))
    mnist.train.images[i] = matrix2vector(tile)

x2,y2 = np.shape(mnist.validation.images)
for i in range(0,x2):
    tile = vector2matrix(mnist.validation.images[i])
    tile = np.dot(permutation_matrix,tile)
    tile = np.dot(tile,np.transpose(permutation_matrix))
    mnist.validation.images[i] = matrix2vector(tile)

x3,y3 = np.shape(mnist.test.images)
for i in range(0,x3):
    tile = vector2matrix(mnist.test.images[i])
    tile = np.dot(permutation_matrix,tile)
    tile = np.dot(tile,np.transpose(permutation_matrix))
    mnist.test.images[i] = matrix2vector(tile)

#画出置换后的图像
plt.figure()
plt.imshow(vector2matrix(mnist.train.images[0]))
plt.show()

plt.figure()
plt.imshow(vector2matrix(mnist.validation.images[0]))
plt.show()

plt.figure()
plt.imshow(vector2matrix(mnist.test.images[0]))
plt.show()

#2.训练过程
# (1).设置输入和输出节点的个数,配置神经网络的参数
INPUT_NODE = 784      # 输入节点个数，对于该数据集就是图片像素总数
OUTPUT_NODE = 10      # 输出节点，等同于类别个数
LAYER1_NODE = 500     # 隐藏层数，此处使用仅有一个隐藏层，500个节点
BATCH_SIZE = 100      # 一个batch中样本个数，数字越小，越接近随机梯度下降，数字越大，越接近梯度下降

# 模型相关的参数
LEARNING_RATE = 0.9      # 基础学习率
REGULARAZTION_RATE = 0.0001   # 描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 100000        # 训练次数


#(2). 定义辅助函数来计算前向传播结果，使用ReLU做为激活函数。
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 不使用滑动平均类
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        # 使用滑动平均类
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)
    
train_loss = []  # 用于保存训练集下loss值
valid_acc = []  # 用于保存验证集下准确率
test_acc = []  # 用于保存测试集下准确率

#(3). 定义训练过程
def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    # 生成隐藏层的参数。
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    # 生成输出层的参数。
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算不含滑动平均类的前向传播结果
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 定义训练轮数及相关的滑动平均类
    global_step = tf.Variable(0, trainable=False)
    average_y = inference(x, None, weights1, biases1, weights2, biases2)

    # 计算交叉熵及其平均值
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 损失函数的计算
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    regularaztion = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularaztion


    # 优化损失函数
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss, global_step=global_step)

    # 反向传播更新参数和更新每一个参数的滑动平均值
    with tf.control_dependencies([train_step]):
        train_op = tf.no_op(name='train')

    # 计算正确率
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # 初始化回话并开始训练过程。
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        # 循环的训练神经网络。
        for i in range(3*TRAINING_STEPS):
            if i % 1000 == 0:
                xs, ys = mnist.train.next_batch(BATCH_SIZE)
                sess.run(train_op, feed_dict={x: xs, y_: ys})
                train_error=sess.run(loss,feed_dict={x: xs, y_: ys})
                train_loss.append(train_error)
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                valid_acc.append(validate_acc)
                test_acc_att= sess.run(accuracy, feed_dict=test_feed)
                test_acc.append(test_acc_att)



#(4). 主程序入口
def main(argv=None): 
    train(mnist)

if __name__=='__main__':
    tf.app.run()
    
#5. 画图
t=np.arange(0,300,1)
plt.ylabel("loss")
plt.xlabel("Periods/every 1000 times ")
plt.title("loss vs. Periods")
fig = plt.gcf()
fig.set_size_inches(13.5, 5.5)
plt.plot(t,train_loss,label="loss")
plt.legend()
plt.show() 

plt.ylabel("validate accuracy")
plt.xlabel("Periods/every 1000 times")
plt.title("validate accuracy vs. Periods")
fig = plt.gcf()
fig.set_size_inches(13.5, 5.5)
plt.plot(t,valid_acc, label="accuracy")
plt.legend()
plt.show() 

plt.ylabel("test accuracy")
plt.xlabel("Periods/every 1000 times")
plt.title("test accuracy vs. Periods")
fig = plt.gcf()
fig.set_size_inches(13.5, 5.5)
plt.plot(t,test_acc, label="accuracy")
plt.legend()
plt.show()     
    