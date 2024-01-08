n,s=8,10		# n和s是全局变量
def fact(n) :		# fact()函数中的n是局部变量和s是全局变量
    global s
    for i in range(1,n+1):  
        s*=i
    return s
print(fact(n),s)	# n和s是全局变量