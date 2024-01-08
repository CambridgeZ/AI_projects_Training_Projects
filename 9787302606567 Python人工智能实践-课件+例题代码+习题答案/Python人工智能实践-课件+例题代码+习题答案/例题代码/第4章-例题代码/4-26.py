import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['axes.unicode_minus'] = False
a = np.arange(0.0,5.0,0.01)
plt.xlabel('时间',fontproperties='SimHei',fontsize=10)
plt.ylabel("振幅",fontproperties='SimHei',fontsize=10)
plt.plot(a,np.cos(2*np.pi*a),'k--')
plt.show()
