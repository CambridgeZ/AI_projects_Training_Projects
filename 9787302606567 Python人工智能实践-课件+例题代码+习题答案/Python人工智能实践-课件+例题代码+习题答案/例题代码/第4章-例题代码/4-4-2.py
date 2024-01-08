fo = open('file.txt',"r",encoding='utf-8')
while txt!=’’:
    txt=fo.read(5)
#处理txt
fo.close()
