import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

classes = 3
labels = tf.constant([1, 0, 2])  # 输入的元素值最小为 0，最大为 2
output = tf.one_hot(labels, depth=classes)
print("result of labels1:", output)
print("\n")
