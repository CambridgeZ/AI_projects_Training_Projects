def func(n,m):
    s=0
    for i in range(1,n+1):
        s+=i
    return s,s/m
s,ave=func(10,10)
print(s, ave)

