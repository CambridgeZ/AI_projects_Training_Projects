# 导入所需库
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd


#从csv文件中读取数据存入x,y
#x:消费周期  y:消费金额
data = pd.read_csv('company.csv')
x=data['period']
y=data['avemoney']
print(x)
print(y)


# 划分3个类别
sortColors = ['g', 'r', 'b']  # 三个类别的三种颜色

# 初始化质心
sortX = np.random.randint(10, 100, 3)  # 生成三个随机数x
sortY = np.random.randint(100, 700, 3)  # 生成三个随机数y

# 初始化类别为0
sortF = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# 3个类别统计，用于迭代更新
sortM = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]  # 分别代表x,y,个数

# k-means算法
Mynum = 10
while (Mynum > 0):
    sortM = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]  # 清零
    flag = 0  # 标记类别
    for i in range(len(x)):
        min = 1000  # 标记最短距离，初始化为1000
        for j in range(3):  # 找出距离最短的坐标
            value = math.sqrt((x[i] - sortX[j]) * (x[i] - sortX[j]) + (y[i] - sortY[j]) * (y[i] - sortY[j]))
            if min > value:  # 判断距离是否是最短
                min = value
                flag = j
                sortM[j][0] += x[i]
                sortM[j][1] += y[i]
                sortM[j][2] += 1
        sortF[i] = flag
    for i in range(3):  # 更新三个质心坐标
        sortX[i] = sortM[i][0] // sortM[i][2]
        sortY[i] = sortM[i][1] // sortM[i][2]
    Mynum -= 1

# 可视化

# 设置中文显示
plt.rcParams['font.sans-serif'] = 'simHei'  # 正确显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

plt.figure()
plt.title('用户分类图')
plt.xlabel("平均消费周期（天）")
plt.ylabel("平均每次消费金额")

# 显示坐标轴元素
plt.xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.yticks([100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800])

# 显示三个中心点
for i in range(3):
    plt.scatter(sortX[i], sortY[i], marker='+', color=sortColors[i], label='1', s=30)

# 显示其他点
for i in range(3):
    for j in range(len(x)):
        if sortF[j] == i:
            plt.scatter(x[j], y[j], marker='.', color=sortColors[i], label='1', s=50)
plt.show()