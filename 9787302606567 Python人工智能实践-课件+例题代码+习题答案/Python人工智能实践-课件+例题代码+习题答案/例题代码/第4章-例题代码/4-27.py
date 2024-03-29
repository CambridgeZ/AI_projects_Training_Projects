import numpy as np
import matplotlib.pyplot as plt

def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

def g(t):
    return np.exp(-t) * np.sin(2*np.pi*t)

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

plt.figure(1)
plt.subplot(221)
plt.plot(t1, f(t1), 'b')

plt.subplot(222)
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')

plt.subplot(223)
plt.plot(t1, g(t1), 'b')

plt.subplot(224)
plt.plot(t2, np.sin(2*np.pi*t2), 'r--')

plt.show()
