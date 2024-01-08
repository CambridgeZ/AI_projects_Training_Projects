score=eval(input())
if score>=90:
    grade='A'
elif score>=80:
    grade='B'
elif score>=70:
    grade='C'
elif score>=60:
    grade='D' 
else:
    grade='E'
print("输入的成绩属于级别{}".format(grade))
