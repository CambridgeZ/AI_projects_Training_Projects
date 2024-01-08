fo = open('file.txt',"r",encoding='utf-8')
for line in fo:
    #分行读入处理
    print(line)
fo.close()
