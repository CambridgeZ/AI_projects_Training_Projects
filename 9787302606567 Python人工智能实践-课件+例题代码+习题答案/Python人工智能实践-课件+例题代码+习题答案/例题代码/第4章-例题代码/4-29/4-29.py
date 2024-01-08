import matplotlib.pyplot as plt

def read_txt(file):
    with open(file, 'r') as temp:
        ls = [x.strip().split(',') for x in temp]
    return ls

data=read_txt('data.txt')

labels =[x[0] for x in data]
sizes = [x[1] for x in data]
print(labels)
print(sizes)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决保存图像是负号'-'显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

explode = [0, 0.1, 0, 0, 0, 0, 0, 0, 0]  # 使python突出显示
plt.axes(aspect=1)
plt.figure(figsize=(10,6.5))
plt.pie(sizes, explode=explode, labels=labels,
        labeldistance=1.1, autopct='%2.1f%%',
        shadow=True, startangle=90,
        pctdistance=0.8)
plt.legend(loc='lower left', bbox_to_anchor=(-0.35, 0.1))
plt.show()
plt.savefig('program.png')

