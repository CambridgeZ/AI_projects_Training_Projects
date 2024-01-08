import csv
with open('rtest.csv', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
          print(', '.join(row))
