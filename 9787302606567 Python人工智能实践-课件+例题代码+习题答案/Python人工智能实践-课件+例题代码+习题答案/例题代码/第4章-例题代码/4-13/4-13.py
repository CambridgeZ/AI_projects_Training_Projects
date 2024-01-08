import json
fr = open("example.csv", "r")
ls = []
for line in fr:
    line = line.replace("\n","")
    ls.append(line.split(','))
fr.close()
print(ls)
fw = open("example.json", "w")
for i in range(1,len(ls)):
    ls[i] = dict(zip(ls[0], ls[i]))
json.dump(ls[1:],fw, sort_keys=True, indent=4)
fw.close()

