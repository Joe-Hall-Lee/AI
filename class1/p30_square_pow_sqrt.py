import tensorflow as tf

a = tf.fill([1, 2], 3.)
print("a:", a)
print("a 的平方:", tf.pow(a, 3))
print("a 的平方:", tf.square(a))
print("a 的开方:", tf.sqrt(a))
