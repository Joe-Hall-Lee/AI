import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

x = tf.Variable(4)
x.assign_sub(1)
print("x:", x)  # 4 - 1 = 3
