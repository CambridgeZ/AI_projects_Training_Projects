import csv
ls=[[1,2,3],['a','b','c']]
with open('wtest.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(ls)
