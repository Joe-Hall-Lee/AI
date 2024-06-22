import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

w = tf.Variable(tf.constant(5, dtype=tf.float32))
lr = 0.2
epoch = 40

for epoch in range(epoch):  # for epoch 定义顶层循环，表示对数据集循环 epoch 次，此例数据集数据仅有 1 个 w，初始化时候 constant 赋值为 5，循环 40 次迭代。
    with tf.GradientTape() as tape:  # with 结构到 grads 框起了梯度的计算过程。
        loss = tf.square(w + 1)
    grads = tape.gradient(loss, w)  # .gradient 函数告知谁对谁求导

    w.assign_sub(lr * grads)  # .assign_sub 对变量做自减 即：w -= lr*grads 即 w = w - lr * grads
    print("After %s epoch,w is %f,loss is %f" % (epoch, w.numpy(), loss))

# lr 初始值：0.2，请自改学习率 0.001～0.999 看收敛过程
# 最终目的：找到 loss 最小 即 w = -1 的最优参数 w
