#coding=utf-8
import tensorflow as tf
import keras_preprocessing.image as k_img
import numpy as np
from matplotlib import pyplot as plt

#图像数据的读入
img = k_img.load_img("sample_cat.jpg") #从文件中读入数据
print("image size:", img.size)#输出图像像素大小(宽, 高)=(500, 750)
img = img.resize((768, 1024))#可以重新调整图像大小
print("image resize:", img.size)#调整后的图像大小(宽, 高)=(768,1024)

img_data = np.array(img) #转为数组(张量)
print("img_data shape", img_data.shape)#输出维数(高, 宽, 通道)=(1024, 768, 3)

img_data = np.expand_dims(img_data, axis=0)#增加样本数(Samples)维度
print("img_data.shape:", img_data.shape)#输出维数(样本数, 高, 宽, 通道)=(1, 1024, 768, 3)

#图像数据的展示
fig1, ax1 = plt.subplots(1,4)#将画板划分为1行4列4张图像
ax1[0].imshow(img_data[0])#展示整个图像
ax1[1].imshow(img_data[0, :, :, 0])#展示红色通道
ax1[2].imshow(img_data[0, :, :, 1])#展示绿色通道
ax1[3].imshow(img_data[0, :, :, 2])#展示蓝色通道
fig1.savefig(fname="image_show.png", format='png')#保存展示结果为图片文件

#线性变换y=2x+8
img_data_01= img_data*2 + 8 #3个通道进行线性变换
img_data_02 = img_data*1;  img_data_02[0, :, :, 0] = img_data_02[0, :, :, 0]*2 + 8 #红色通道进行线性变换
img_data_03 = img_data_02*1; img_data_03[0, :, :, 1] = img_data_03[0, :, :, 1]*2 + 8 #绿色通道再进行线性变换
img_data_04 = img_data_03*1; img_data_04[0, :, :, 2] = img_data_04[0, :, :, 2]*2 + 8 #绿色通道再进行线性变换
fig2,ax2 = plt.subplots(1,5)
ax2[0].imshow(img_data[0]) #原始图像
ax2[1].imshow(img_data_01[0]) #RGB3通道同时变换
ax2[2].imshow(img_data_02[0]) #变换红色通道
ax2[3].imshow(img_data_03[0]) #再变换绿色通道
ax2[4].imshow(img_data_04[0]) #再变换蓝色通道
fig2.savefig(fname="image_linear.png", format='png')#保存结果为图片文件

#几何变换
img_data_11 = k_img.random_shift(img_data[0], 0.2, 0.3, row_axis=0, col_axis=1, channel_axis=2)
img_data_12 = k_img.random_zoom(img_data[0], (0.5,0.5), row_axis=0, col_axis=1, channel_axis=2)
img_data_13 = k_img.random_shear(img_data[0], 60,  row_axis=0, col_axis=1, channel_axis=2)
img_data_14 = k_img.random_rotation(img_data[0], 90,  row_axis=0, col_axis=1, channel_axis=2)
fig3, ax3 = plt.subplots(1, 5)
ax3[0].imshow(img_data[0])
ax3[1].imshow(img_data_11)
ax3[2].imshow(img_data_12)
ax3[3].imshow(img_data_13)
ax3[4].imshow(img_data_14)
fig3.savefig(fname="image_geometric.png", format='png')
