#file LeNet_MNIST.py
#coding=utf-8

import tensorflow as tf
from matplotlib import pyplot as plt
# from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical 
from keras import models
from keras import layers
import keras_preprocessing.image as k_img
# import tensorflow.keras.preprocessing.image as k_img
import numpy as np

#输入层大小
INPUT_SHAPE = (28,28,1)
#第一个卷积层的卷积核大小和数量
CONV1_SIZE = 5
CONV1_NUM = 6
#第二个卷积层的卷积核大小和数量
CONV2_SIZE = 5
CONV2_NUM = 16
#池化层窗口大小
POOL_SIZE = 2
#全连接层节点个数
FC1_SIZE = 120
FC2_SIZE = 84
#输出个数
OUT_SIZE = 10
#训练参数
EPOCH_SIZE = 20
BATCH_SIZE = 200

#加载和预处理数据
def load_images_data():
    #加载图像和标签数据
    (train_images,train_labels),(test_images,test_labels) = tf.keras.datasets.mnist.load_data()   
    print("train_images:", train_images.shape)
    print("train_labels:", train_labels.shape)
    print("test_images:", test_images.shape)
    print("test_labels:", test_labels.shape)

    assert train_images.shape==(60000,28,28), "load train images error!"
    assert train_labels.shape==(60000,), "load train labels error!"
    assert test_images.shape==(10000,28,28), "load test images error!"
    assert test_labels.shape==(10000,), "load test labels error!"

    #预处理数据
    N0 = train_images.shape[0];
    N1 = test_images.shape[0]
    print(N0,N1)
    
    train_images = train_images.reshape(N0,28,28,1)
    train_images = train_images.astype('float32') / 255
    train_labels = to_categorical(train_labels)

    test_images = test_images.reshape(N1,28,28,1)
    test_images = test_images.astype('float32') / 255
    test_labels = to_categorical(test_labels)

    return train_images,train_labels,test_images,test_labels

#创建LeNet5网络
def build_LeNet5():
    model = models.Sequential()
    #第一层：卷积层
    model.add(layers.Conv2D(filters=CONV1_NUM,kernel_size=(CONV1_SIZE,CONV1_SIZE),padding="same",activation='relu',input_shape=INPUT_SHAPE,name="layer1-conv1"))
    #第二层：最大池化层
    model.add(layers.MaxPooling2D(pool_size=(POOL_SIZE,POOL_SIZE),name="layer2-pool"))
    #第三层：卷积层
    model.add(layers.Conv2D(filters=CONV2_NUM,kernel_size=(CONV2_SIZE,CONV2_SIZE),padding="valid",activation='relu',name="layer3-conv2"))
    #第四层：最大池化层
    model.add(layers.MaxPooling2D(pool_size=(POOL_SIZE,POOL_SIZE),name="layer4-pool"))
    model.add(layers.Flatten(name="layer4-flatten"))
    #第五层：全连接层
    model.add(layers.Dense(units=FC1_SIZE,activation='relu',name="layer5-fc1"))
    model.add(layers.Dense(units=FC2_SIZE,activation='relu',name="layer5-fc2"))
    #第六层：softmax输出层
    model.add(layers.Dense(units=OUT_SIZE,activation='softmax',name="layer6-fc"))

    return model

#模型训练
def train_LeNet5(model,train_data,train_labels,test_data,test_labels):
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    history = model.fit(x=train_data,y=train_labels,epochs=EPOCH_SIZE,batch_size=BATCH_SIZE,validation_data=[test_data,test_labels])

    return history

#绘制loss和accuracy
def draw_history(history):

    loss = history.history['loss']
    accuracy = history.history['accuracy']
    val_loss = history.history['val_loss']
    val_accuracy = history.history['val_accuracy']
    epochs = range(1, len(loss) + 1)

    #draw loss with epoch
    plt.subplot(2,2,1)
    plt.plot(epochs,loss,'bo')
    plt.title("Training loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.legend()

    #draw accuracy with epoch
    plt.subplot(2,2,2)
    plt.plot(epochs,accuracy,'bo')
    plt.title("Training accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    #plt.legend()

    #draw val_loss with epoch
    plt.subplot(2,2,3)
    plt.plot(epochs,val_loss,'bo')
    plt.title("Validate loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.legend()

    #draw val_accuracy with epoch
    plt.subplot(2,2,4)
    plt.plot(epochs,val_accuracy,'bo')
    plt.title("Validate accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    #plt.legend()

    plt.tight_layout()
    plt.show()

    #save to file
    plt.savefig(fname="LetNet5-history.png",format='png')

#模型预测
def use_LeNet5(img_file, model):
    x0 = k_img.load_img(img_file)
    print("image size:", x0.size)

    #适配LetNet5输入大小
    x1 = x0.resize((28,28))
    #转为数组
    x1 = np.array(x1)
    print("x1.shape:", x1.shape)

    #转为灰度图
    x2 = tf.image.rgb_to_grayscale(x1)
    #不能转为2维图片
    #x2 = tf.squeeze(x2, 2)
    print("x2.shape:", x2.shape)
    
    #转为LeNet-5的输入维度(NONE,28,28,1)
    #x3 = x2.reshape(1,28,28,1)
    x3 = np.expand_dims(x2, axis=0)
    print("x3.shape:", x3.shape)

    #预测结果
    y = model.predict(x3)
    print("predict result:", y)

    #可视化输出图片和预测结果
    plt.figure()
    fig,ax = plt.subplots(1,4)
    ax[0].imshow(x0)
    ax[1].imshow(x1)
    ax[2].imshow(x2[:,:,0],cmap='gray')
    plt.subplot(1,4,4)
    plt.plot(range(0,10), y[0],'bo')
    plt.tight_layout()

    plt.show()

    #save to file
    plt.savefig(fname="LetNet5-predict.png",format='png')

    return y


#主流程
train_images,train_labels,test_images,test_labels =load_images_data()
model = build_LeNet5()
model.summary()
history = train_LeNet5(model,train_images,train_labels,test_images,test_labels)
draw_history(history)
label = use_LeNet5("mydigit.jpg", model)


