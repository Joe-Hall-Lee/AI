import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

x = tf.constant([[1, 2, 3], [2, 2, 3]])
print("x:", x)
print("mean of x:", tf.reduce_mean(x))  # 求 x 中所有数的均值
print("sum of x:", tf.reduce_sum(x, axis=1))  # 求每一行的和
