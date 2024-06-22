import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

a = tf.ones([1, 3])
b = tf.fill([1, 3], 3.)
print("a:", a)
print("b:", b)
c = a + b
print("a+b:", tf.add(a, b))
d = a - b
print("a-b:", tf.subtract(a, b))
e = a * b
print("a*b:", tf.multiply(a, b))
f = b / a
print("b/a:", tf.divide(b, a))
