fo = open("output.txt","w+")
ls = ["Python语言", "Java语言", "C语言"]
fo.writelines(ls)
fo.seek(0)
for line in fo:
    print(line)
fo.close()