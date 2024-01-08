# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 10:34:39 2022

@author: bruce dee
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据集
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
learning_rate = 0.01
max_samples = 40000
batch_size = 128
 
n_steps = 28
n_inputs = 28
n_hidden = 256
n_classes = 10
 
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])
 
weights = tf.random_normal([2 * n_hidden, n_classes])
biases = tf.random_normal([n_classes])
 
 
def BiRNN(x, weights, biases):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_inputs])
    x = tf.split(x, n_steps)
 
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
    y_ = tf.matmul(outputs[1], weights["weight_out"]) + biases["biases_out"]
    return y_
 
 
prediction = BiRNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
 
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1)), tf.float32))
 
init = tf.global_variables_initializer()
 
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < max_samples:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, n_steps, n_inputs))
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % 10 == 0:
            accuracy = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            print(accuracy)
        step += 1
 
    x_batch = mnist.test.images[:1000].reshape((-1, n_steps, n_inputs))
    y_batch = mnist.test.labels[:1000]
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: x_batch, y: y_batch}))