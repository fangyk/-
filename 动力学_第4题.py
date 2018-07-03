# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 10:48:20 2018

@author: lenovo
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.framework import ops
import random
import struct
from sklearn.datasets import fetch_lfw_people
import math
#1.读取lfw数据，裁剪成80*80大小 
lfw=fetch_lfw_people(data_home=None,resize=0.9)
n_samples,h,w=lfw.images.shape
lfw.images=lfw.images[0:13233,16:96,2:82]#共13233幅人脸图片，每幅大小为80*80
choose_images_as_train = random.sample(range(0,13233,1),10000)#选出10000张作为训练集，3233张作为验证集
lfw_train_images = lfw.images[choose_images_as_train]
lfw_validation_images = lfw.images[np.delete(range(0,13233,1),choose_images_as_train)]
'''
#画图
plt.figure()
plt.imshow(lfw_train_images[0])
plt.show()
'''

#2.随机置换矩阵
def swapRows(M, r1, r2):  
    M[r1],M[r2] = M[r2],M[r1]  
    return(M) 

def random_permutation(M):
    N = random.sample(range(1,8,1),1)#随机置换的次数
    row_change = [round(random.uniform(0,7)) for i in range(N[0])]#随机置换的行
    permutation_matrix=np.eye(8)
    for i in range(0,N[0]-1):
        permutation_matrix=swapRows(permutation_matrix,row_change[i],row_change[i+1])
    #80*80的矩阵平均分成8*8块，每一块的大小为10*10
    p_matrix = np.eye(80)-np.eye(80)
    for i in range(0,8,1):
        for j in range(0,8,1):
            if permutation_matrix[i,j]==1 :
                for k in range(0,10,1):
                    p_matrix[10*i+k,10*j+k]=1.0
    M=np.dot(p_matrix,M)
    M=np.dot(M,np.transpose(p_matrix))            
    return(M)
'''   
#画出置换后的图片     
lfw_train_images[0]        
random_permutation(lfw_train_images[0])     
plt.figure()
plt.imshow(random_permutation(lfw_train_images[0]))
plt.show()   
'''           
#3.将训练集进行置换,每张图片置换五次
lfw_p_train_images=np.random.rand(30000,80,80)

for k in range(0,10000,1):
    lfw_p_train_images[3*k]=random_permutation(lfw_train_images[k])
    lfw_p_train_images[3*k+1]=random_permutation(lfw_train_images[k])
    lfw_p_train_images[3*k+2]=random_permutation(lfw_train_images[k])
                                   
#4.改变数据集标签
#读入的训练集所有标签都改为（1,0）,表示的为是人脸图片
#将训练集进行随机置换，置换后的数据的标签为（0,1）,表示的为不是人脸图片
lfw_train_images=np.reshape(lfw_train_images,[10000,6400])
lfw_train_labels=[[1,0]]*10000
lfw_p_train_images=np.reshape(lfw_p_train_images,[30000,6400])
lfw_p_train_labels=[[0,1]]*30000
lfw_train_images_total=np.vstack((lfw_train_images,lfw_p_train_images))
lfw_train_labels_total=np.vstack((lfw_train_labels,lfw_p_train_labels))  

#5.利用神经网络构建一个二分类模型，输入为未置换的和置换后的数据集，对应标签为（1,0）和（0,1）
# (1).设置输入和输出节点的个数,配置神经网络的参数
INPUT_NODE = 6400      # 输入节点个数，对于该数据集就是图片像素总数
OUTPUT_NODE = 2      # 输出节点，等同于类别个数
LAYER1_NODE = 500     # 隐藏层数，此处使用仅有一个隐藏层，500个节点
BATCH_SIZE = 100      # 一个batch中样本个数，数字越小，越接近随机梯度下降，数字越大，越接近梯度下降

# 模型相关的参数
LEARNING_RATE = 0.9      # 基础学习率
REGULARAZTION_RATE = 0.0001   # 描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 3000        # 训练次数


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
w1=[]
b1=[]
w2=[]
b2=[]


def next_batch(train_data, train_target, batch_size):  
    index = [ i for i in range(0,len(train_target)) ]  
    np.random.shuffle(index);  
    batch_data = []; 
    batch_target = [];  
    for i in range(0,batch_size):  
        batch_data.append(train_data[index[i]]);  
        batch_target.append(train_target[index[i]])  
    return batch_data, batch_target 

#(3). 定义训练过程
def train(lfw):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    # 生成隐藏层的参数。
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1),name='w11')
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]),name='b11')
    # 生成输出层的参数。
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1),name='w22')
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]),name='b22')

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
    saver=tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        


        # 循环的训练神经网络。
        for i in range(3*TRAINING_STEPS):
                xs, ys = next_batch(lfw_train_images_total,lfw_train_labels_total,BATCH_SIZE)
                sess.run(train_op, feed_dict={x: xs, y_: ys})
                train_error=sess.run(loss,feed_dict={x: xs, y_: ys})
                train_loss.append(train_error)
               
        saver.save(sess,"G:/课程作业/讨论班/model.ckpt")


#(4). 主程序入口
def main(argv=None): 
    train(lfw)

if __name__=='__main__':
    tf.app.run()                  


#6.对置换后的验证集，进行随机置换回去，所得的数据经过5所建立的二分类模型，当判断是人脸时就输出置换回去的矩阵
#当判断不是人脸时，继续做置换，直至满足判断为人脸的结果为止
#利用输出的矩阵和原来验证集的矩阵进行比较，得到准确率
#读取权重和偏置
saver=tf.train.import_meta_graph("G:/课程作业/讨论班/model.ckpt.meta") 
with tf.Session() as sess:
    saver.restore(sess,"G:/课程作业/讨论班/model.ckpt")
    w1=sess.run(tf.get_default_graph().get_tensor_by_name("w11:0"))
    w2=sess.run(tf.get_default_graph().get_tensor_by_name("w22:0"))
    b1=sess.run(tf.get_default_graph().get_tensor_by_name("b11:0"))
    b2=sess.run(tf.get_default_graph().get_tensor_by_name("b22:0"))
    
#得到随机置换矩阵    
lfw_p_validation_images=np.random.rand(3233,80,80)
for k in range(0,3233,1):
    lfw_p_validation_images[k]=random_permutation(lfw_validation_images[k])
    
#置换回原来矩阵
validation_images=np.random.rand(3233,6400)
validation_labels_get=np.random.rand(3233,2)
loop=30000#最多搜索30000次
begin=0    
def get_value(M,w1,b1,w2,b2):
    layer1 = np.dot(M,w1)+b1
    x1=np.shape(layer1)
    for i in range(0,x1[0],1):
        if layer1[i]<0:
            layer1[i]=0
    layer2=np.dot(layer1,w2)+b2
    z1=layer2[0]
    z2=layer2[1]
    if z1<z2:
        return(1)
    else:
        return(0)
        
        
for k in range(0,3233,1):
    while begin < loop:
        validation_images[k]=np.reshape(random_permutation(lfw_p_validation_images[k]),6400)    
        labels=get_value(validation_images[k],w1,b1,w2,b2)        
        if labels==0:
            validation_images[k]=validation_images[k]
            break
        begin = begin+1

#验证集的准确率
acc=np.random.rand(3233)
lfw_validation_images=np.reshape(lfw_validation_images,[3233,6400])
for k in range(0,3233,1): 
    s=0
    for i in range(0,6400,1):
        if lfw_validation_images[k][i]==validation_images[k][i]:
            s=s+1
    acc[k]=s/6400

np.shape(acc[acc==0])

    











