n,s=8,10		# n和s是全局变量
def fact(n) :		# fact()函数中的n和s是局部变量
    s=1
    for i in range(1,n+1):
        s*=i
    return s 
print(fact(n),s)	# 此处的n和s是全局变量
