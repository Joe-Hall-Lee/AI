from PIL import Image
import numpy as np
import tensorflow as tf

# 模型保存路径
model_save_path = './checkpoint/mnist.ckpt'

# 构建模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),  # 指定输入形状
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 加载模型权重
model.load_weights(model_save_path)

# 获取测试图片数量
preNum = int(input("请输入测试图片的数量："))

for i in range(preNum):
    image_path = input("请输入测试图片的路径：")

    # 打开并处理图片
    img = Image.open(image_path)
    img = img.resize((28, 28), Image.LANCZOS)  # 使用 LANCZOS 进行高质量缩放
    img_arr = np.array(img.convert('L'))  # 转换为灰度图

    # 反转颜色
    img_arr = 255 - img_arr

    # 归一化
    img_arr = img_arr / 255.0
    print("图片数组形状:", img_arr.shape)

    # 增加一个维度以符合模型输入
    x_predict = img_arr[tf.newaxis, ...]
    print("预测输入形状:", x_predict.shape)

    # 进行预测
    result = model.predict(x_predict)

    # 获取预测结果
    pred = tf.argmax(result, axis=1)

    # 打印预测结果
    print('\n预测结果:')
    tf.print(pred)
