from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
type = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model_save_path = './checkpoint/fashion.ckpt'
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.build((None, 28, 28))
model.load_weights(model_save_path)

preNum = int(input("input the number of test pictures:"))
for i in range(preNum):
    image_path = input("the path of test picture:")

    img = Image.open(image_path)

    image = plt.imread(image_path)
    plt.set_cmap('gray')
    plt.imshow(image)

    img=img.resize((28,28),Image.ANTIALIAS)
    img_arr = np.array(img.convert('L'))
    img_arr = 255 - img_arr

    img_arr=img_arr/255.0

    x_predict = img_arr[tf.newaxis,...]

    result = model.predict(x_predict)
    pred=tf.argmax(result, axis=1)
    print('\n')
    print(type[int(pred)])

    plt.pause(1)
    plt.close()