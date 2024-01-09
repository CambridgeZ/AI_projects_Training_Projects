#file cnn_net.py
#coding=utf-8
import tensorflow as tf
from keras import models
from keras import layers

#LeNet5
def LeNet_5():
    model = models.Sequential()
    model.add(layers.Conv2D(filters=6,kernel_size=(5,5),padding="same",activation='relu',input_shape=(28,28,1)))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Conv2D(filters=16,kernel_size=(5,5),padding="valid",activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=120,activation='relu'))
    model.add(layers.Dense(units=84,activation='relu'))
    model.add(layers.Dense(units=10,activation='softmax'))

    model.summary()

    return model

#Alex Net
def AlexNet():
    model = models.Sequential()

    model.add(layers.Conv2D(filters=96,kernel_size=(11,11),strides=(4,4),input_shape=(227,227,3),padding='valid',activation='relu',kernel_initializer='uniform'))
    model.add(layers.MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(layers.Conv2D(filters=256,kernel_size=(5,5),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(layers.MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(layers.Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(layers.Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(layers.Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(layers.MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=4096,activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(units=4096,activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(units=1000,activation='softmax'))
    
    model.summary()
    
    return model

#VGG-16 Net
def VGGNet_16():   
    model = models.Sequential()
    
    model.add(layers.Conv2D(64,(3,3),strides=(1,1),input_shape=(224,224,3),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(layers.Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))    
    model.add(layers.Conv2D(128,(3,2),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(layers.Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))    
    model.add(layers.Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(layers.Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(layers.Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))    
    model.add(layers.Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(layers.Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(layers.Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))    
    model.add(layers.Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(layers.Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(layers.Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))    
    model.add(layers.Flatten())
    model.add(layers.Dense(4096,activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096,activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1000,activation='softmax'))
    
    model.summary()
    
    return model

#Inception_v1 used in googlenet
def inception_v1(x,f1,f2_1,f2_2,f3_1,f3_2,f4):
    #branch1（第1路）:1x1
    b1 = layers.Conv2D(filters=f1, kernel_size=1, activation='relu')(x)

    #branch2（第2路）:3x3
    b2 = layers.Conv2D(filters=f2_1, kernel_size=1, activation='relu')(x)
    b2 = layers.Conv2D(filters=f2_2, kernel_size=3, activation='relu', padding="same")(b2)

    #branch3（第3路）:5x5
    b3 = layers.Conv2D(filters=f3_1, kernel_size=1, activation='relu')(x)
    b3 = layers.Conv2D(filters=f3_2, kernel_size=5,activation='relu', padding="same")(b3)

    #branch4（第4路）:pool
    b4 = layers.MaxPooling2D(pool_size=(3,3), strides=1, padding="same")(x)
    b4 = layers.Conv2D(filters=f4, kernel_size=1, activation='relu')(b4)

    #4路串联
    out = tf.concat([b1, b2, b3, b4], 3)

    return out

#GoogLeNet
def googLeNet():
    x = layers.Input(shape=(224,224,3))
    #第一组卷积层
    conv1 = layers.Conv2D(filters=64, kernel_size=7, strides=2, activation='relu', padding="same", name="conv1")(x)
    #第一个池化层
    pool1 = layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same", name="pool1")(conv1)
    #第二组卷积层
    conv2_1 = layers.Conv2D(filters=64, kernel_size=1, strides=1, activation='relu', padding="same", name="conv2_1")(pool1)
    conv2_2 = layers.Conv2D(filters=192, kernel_size=3, strides=1, activation='relu', padding="same", name="conv2_2")(conv2_1)
    #第二个池化层
    pool2 = layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same", name="pool2")(conv2_2)
    #第一组Inception层
    incpt1_1 = inception_v1(pool2, 64, 96, 128, 16, 32, 32)
    incpt1_2 = inception_v1(incpt1_1, 128, 128, 192, 32, 96, 64)
    #第三个池化层
    pool3 =layers.MaxPooling2D(pool_size=(3,3), strides=2, name="pool3")(incpt1_2)
    #第二组Inception层
    incpt2_1 = inception_v1(pool3, 192, 96, 208, 16, 48, 64)
    incpt2_2 = inception_v1(incpt2_1, 160, 112, 224, 24, 64, 64)
    incpt2_3 = inception_v1(incpt2_2, 128, 128, 256, 24, 64, 64)
    incpt2_4 = inception_v1(incpt2_3, 112, 144, 288, 32, 64, 64)
    incpt2_5 = inception_v1(incpt2_4, 256, 160, 320, 32, 128, 128)
    #第四个池化层
    pool4 =layers.MaxPooling2D(pool_size=(3,3), strides=2, name="pool4")(incpt2_5)
    #第三组Inception层
    incpt3_1 = inception_v1(pool4, 256, 160, 320, 32, 128, 128)
    incpt3_2 = inception_v1(incpt3_1, 384, 192, 384, 48, 128, 128)
    #第五个池化层
    pool5 = layers.GlobalAveragePooling2D(name="pool5")(incpt3_2)
    flatten = layers.Flatten(name="flatten")(pool5)
    #全连接层
    fc = layers.Dense(units=1000, activation='relu', name="fc")(flatten)
    #输出层
    out = layers.Dense(units=1000, activation='softmax', name="out")(fc)

    model = models.Model(x, out, name="googLeNet")
    model.summary()
 
    return model


#main
#model = LeNet_5()
#model = AlexNet()
#model = VGGNet_16()
model = googLeNet()
#model.summary()