import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.rcParams['font.family']='SimHei'
matplotlib.rcParams['font.size']=10
plt.rcParams['axes.unicode_minus'] = False
a = np.arange(0.0,5.0,0.01)
plt.xlabel('时间')
plt.ylabel("振幅")
plt.plot(a,np.cos(2*np.pi*a),'k--')
plt.show()
