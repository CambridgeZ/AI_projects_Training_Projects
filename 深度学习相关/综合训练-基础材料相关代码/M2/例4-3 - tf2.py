import os
os.environ["TF_KERAS"] = '1'
import tensorflow.compat.v1 as tf
import tensorflow as tf2

tf.disable_v2_behavior()
import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt


def run(x_data, y_data):
    global pred, xs, ys
    xs = tf.placeholder(tf2.float32, [None, x_data.shape[-1]])
    ys = tf.placeholder(tf2.float32, [None, 10])

    pred = tf.layers.dense(inputs=xs, units=10, activation=tf2.nn.softmax)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=pred)


train_step = tf.train.GradientDescentOptimizer(.3).minimize(cross_entropy)

se.run(tf.global_variables_initializer())
results = []
for i in range(2000):
    random_index = np.random.choice(x_train.shape[0], 10, replace=False)
    batch_xs, batch_ys = x_train[random_index], y_train[random_index]

    se.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})

    if i % 50 == 0:
        acc = compute_accuracy(x_test, y_test)
        results.append((i, acc))
        print(acc)
results = np.array(results)
plt.scatter(results[:, 0], results[:, 1])
plt.show()


def compute_accuracy(v_xs, v_ys):
    y_pre = se.run(pred, feed_dict={xs: v_xs})
    correct_pred = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf2.float32))
    result = se.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], -1) / 255
    x_test = x_test.reshape(x_test.shape[0], -1) / 255
    se = tf.Session()

    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    run(x_test, y_test)
