import tensorflow as tf
import math
import numpy as np

# for testing

data = tf.Variable([[1, 2], [3, 2], [3, 5], [7, 1], [8, 9]])
updates = tf.constant([[-1, -1], [-1, -1]])

shape = data.get_shape()


sess = tf.Session()
sess.run(tf.initialize_all_variables())

print sess.run([data])
data2 = tf.reshape(data, [-1])
data3 = tf.tile(data2, tf.pack([3]))
data4 = tf.reshape(data3,[3, -1])
data5 = tf.transpose(data4, [1, 0])
data6 = tf.reshape(data5, [-1, 2, 3])
print sess.run([data6])


