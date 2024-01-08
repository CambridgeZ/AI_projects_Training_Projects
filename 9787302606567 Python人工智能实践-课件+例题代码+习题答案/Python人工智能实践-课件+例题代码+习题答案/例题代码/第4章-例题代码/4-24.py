import matplotlib.pyplot as plt
import numpy as np
a = np.arange(1,11)
plt.plot(a,a*2,'r-',a,a*3,'bx',a,a*4,'g*:',a,a*5,'c-.')
plt.show()
