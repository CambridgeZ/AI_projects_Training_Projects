import math
n=int(input())
for i in range(2,n):
    if n%i==0:
        print("不是素数")
        break
else:
	print("是素数")