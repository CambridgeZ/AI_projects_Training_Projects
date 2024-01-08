import numpy as np
a=np.random.rand(5,3)
print("矩阵为:")
print(a)  #输出5行3列的矩阵
print("第3行为:")
print(a[2])  #输出第3行
print("第2,3行为:")
print(a[1:3])  #输出第2,3行
print("第1,3行为:")
print(a[0:3:2])  #输出第1,3行
