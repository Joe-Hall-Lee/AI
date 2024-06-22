import tensorflow as tf

a = tf.constant([1, 2, 3, 1, 1])
b = tf.constant([0, 1, 3, 4, 5])
c = tf.where(tf.greater(a, b), a, b)  # 若 a > b，返回 a 对应位置的元素，否则返回 b 对应位置的元素
print("c：", c)

