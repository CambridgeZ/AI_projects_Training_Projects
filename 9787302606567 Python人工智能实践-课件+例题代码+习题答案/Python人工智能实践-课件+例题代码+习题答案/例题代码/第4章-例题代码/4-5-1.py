fo = open('file.txt',"r",encoding='utf-8')
lines= fo.readlines()
for line in lines:
    #分行处理
    print(line)
fo.close()
