# coding: utf-8

'''
    Time: 2018-10-06    Author: Snowy
    Tensorflow practice 1： 对评论进行分类

'''

import numpy as np
import tensorflow as tf
import random
import pickle
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize

import logging

"""
'I am super man'
tokenize: ['I', 'am', 'super', 'man']
"""

from nltk.stem import WordNetLemmatizer
"""
词形还原（lemmatizer），即把一个任何形式的英语单词还原到一般形式，与词根还原不同(stemmer)
"""

pos_file = "pos.txt"
neg_file = "neg.txt"


# 读取文件
def process_file(fileName):
    with open(fileName , 'r', encoding="utf-8") as f:
        lex = []
        lines = f.readlines()

        for line in lines:
            # 进行分词
            words = word_tokenize(line.lower())
            lex += words
        return lex



# 创建词汇表
def create_lexicon(pos_file, neg_file):
    lex = []

    lex += process_file(pos_file)
    lex += process_file(neg_file)
    print (len(lex))


    lemmatizer = WordNetLemmatizer()
    # 词形还原 (cats => cat)
    lex = [lemmatizer.lemmatize(word) for word in lex]

    word_count = Counter(lex)
    print (word_count)
    # {'.': 13944, ',': 10536, 'the': 10120, 'a': 9444, 'and': 7108, 'of': 6624, 'it': 4748, 'to': 3940......}
    # 去掉一些常用词,像the,a and等等，和一些不常用词; 这些词对判断一个评论是正面还是负面没有做任何贡献
    lex = []
    for word in word_count:
        if word_count[word] < 2000 and word_count[word] > 10:
            lex.append(word)

    return lex


'''
    把每条评论转换为向量, 转换原理：
    假设lex为['woman', 'great', 'feel', 'actually', 'looking', 'latest', 'seen', 'is'] 当然实际上要大的多
    评论'i think this movie is great' 转换为 [0,1,0,0,0,0,0,1], 把评论中出现的字在lex中标记，出现过的标记为1，其余标记为0
'''

'''
    :param lex:词汇表
    :param review:评论
    :param clf:评论对应的分类，[0,1]代表负面评论 [1,0]代表正面评论
'''
def string_to_vector(lex, review, clf):
    words = word_tokenize(review.lower())
    lemmatizer = WordNetLemmatizer()
    words = [ lemmatizer.lemmatize(word) for word in words]

    features = np.zeros(len(lex))
    for word in words:
        if word in lex:
            features[lex.index(word)] = 1

    return [features, clf]


def normalize_dataset(lex):
    dataset = []

    with open(pos_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            one_sample = string_to_vector(lex, line, [1, 0])
            dataset.append(one_sample)
    with open(neg_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            one_sample = string_to_vector(lex, line, [0, 1])
            dataset.append(one_sample)

    return dataset

# 获取词汇表
lex = create_lexicon(pos_file, neg_file)
# 将样本数据转换成向量格式
dataset = normalize_dataset(lex)

print (len(dataset))

# 使得dataset随机排序
random.shuffle(dataset)

'''
    #把整理好的数据保存到文件，方便使用。到此完成了数据的整理工作
'''
with open('save.pickle', 'wb') as f:
    pickle.dump(dataset, f)


# 取样本中的10%作为测试数据
test_size = int(len(dataset) * 0.1)

dataset = np.array(dataset)

train_dataset = dataset[: -test_size]
test_dataset = dataset[-test_size:]


##############################################################################
'''
    神经网络部分
'''
##############################################################################

# Feed-Forword Neural Network
# 定义每个层有多少‘神经元’

# 输入层
n_input_layer = len(lex)

# 第一层 hide layer
n_layer_1 = 2000
# 第二层 hide layer
n_layer_2 = 2000
# 第三层 hide layer
n_layer_3 = 2000

# 输出层
n_output_layer = 2


# 定义待训练的神经网络
def neural_network(data):
    # 定义第一层神经元的权重和biases
    layer_1_w_b = {'w_': tf.Variable(tf.random_normal([n_input_layer, n_layer_1])), 'b_': tf.Variable(tf.random_normal([n_layer_1]))}
    # 定义第二层神经元的权重和biases
    layer_2_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_1, n_layer_2])), 'b_': tf.Variable(tf.random_normal([n_layer_2]))}
    # 定义第二层神经元的权重和biases
    layer_3_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_2, n_layer_3])),
                   'b_': tf.Variable(tf.random_normal([n_layer_3]))}
    # 定义输出层神经元的权重和biases
    layer_output_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_3, n_output_layer])), 'b_': tf.Variable(tf.random_normal([n_output_layer]))}

    # w * x + b
    layer_1 = tf.add(tf.matmul(data, layer_1_w_b['w_']), layer_1_w_b['b_'])
    layer_1 = tf.nn.relu(layer_1)       # 激活函数
    layer_2 = tf.add(tf.matmul(layer_1, layer_2_w_b['w_']), layer_2_w_b['b_'])
    layer_2 = tf.nn.relu(layer_2)  # 激活函数
    layer_3 = tf.add(tf.matmul(layer_2, layer_3_w_b['w_']), layer_3_w_b['b_'])
    layer_3 = tf.nn.relu(layer_3)  # 激活函数

    layer_output = tf.add(tf.matmul(layer_3, layer_output_w_b['w_']), layer_output_w_b['b_'])

    return layer_output


# 每次使用50条数据进行训练
batch_size = 50

X = tf.placeholder('float', [None, len(train_dataset[0][0])])
#[None, len(train_x)]代表数据数据的高和宽（矩阵），好处是如果数据不符合宽高，tensorflow会报错，不指定也可以。
Y = tf.placeholder('float')
# 使用数据训练神经网络

def train_neural_network(X, Y):
    predict = neural_network(X)
    # 损失函数
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))
    # learning rate 默认 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(cost_func)

    epochs = 50

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        epoch_loss = 0

        i = 0
        random.shuffle(train_dataset)
        train_x = train_dataset[:, 0]
        train_y = train_dataset[:, 1]
        # train_x = dataset[:, 0]
        # train_y = dataset[:, 1]

        for epoch in range(epochs):
            while i < len(train_x):
                start = i
                end = i + batch_size

                batch_x = train_x[start: end]
                batch_y = train_y[start: end]

                _, c = sess.run([optimizer, cost_func], feed_dict={X: list(batch_x), Y: list(batch_y)})

                epoch_loss += c

                i += batch_size
            print( epoch, ": ", epoch_loss)

        test_x = test_dataset[:, 0]
        test_y = test_dataset[:, 1]

        correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('准确率: ', accuracy.eval({X: list(test_x), Y: list(test_y)}))

train_neural_network(X, Y)