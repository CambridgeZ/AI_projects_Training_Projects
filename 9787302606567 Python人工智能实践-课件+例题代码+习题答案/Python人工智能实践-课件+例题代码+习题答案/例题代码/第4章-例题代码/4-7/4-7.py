fp=open("grade.csv","r")
ls=[]
for line in fp:
     line=line.replace('\n','')
     ls.append(line.split(','))
print(ls)
fp.close()
