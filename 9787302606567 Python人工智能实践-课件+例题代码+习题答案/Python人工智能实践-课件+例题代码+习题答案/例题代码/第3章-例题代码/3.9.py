ls=["a","b"]	  #通过使用[ ]真实创建了一个全局变量列表ls，内有两个元素
def	func(a) : 
    ls=[] 	  #此处ls是局部变量，是真实创建的空列表 
    ls.append(a) 	#局部变量ls被修改
    print(ls)
func("c")  	  
print(ls)				#此处输出的ls是全局变量，内容仍然为['a', 'b']
