from sklearn import linear_model
import pandas as pd

#1.导入数据
#X：每一项表示年龄和年收入
#y:表示是否买车（0：不买，1：买）
data = pd.read_csv('data.csv')
X=data[['age','income']]
y=data['target']
print('X:')
print(X)
print('y:')
print(y)


#2.逻辑回归，拟合

lr = linear_model.LogisticRegression()
lr.fit(X,y)

#3.预测29岁，年收入在8万，是否会买车
testX = [[29,8]]

#预测并输出预测标签
label = lr.predict(testX)
print("predicted label = ", label)

#输出预测概率
prob = lr.predict_proba(testX)
print("probability = ",prob)
