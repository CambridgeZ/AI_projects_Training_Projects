import matplotlib.pyplot as plt
plt.plot([2,4,6,8,10],[3,2,4,1,5])
plt.xlabel('id')
plt.ylabel('grade')
plt.savefig('figtest',dpi=600)
plt.show()
