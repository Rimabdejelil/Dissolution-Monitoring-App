import csv
import math

with open('Data.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    data = [row for row in reader]

for row in data:
    row['timestamp'] = round(float(row['timestamp']))

for row in data:
    x1, y1, x2, y2 = float(row['x1']), float(row['y1']), float(row['x2']), float(row['y2'])
    area = math.fabs((x2 - x1) * (y2 - y1))
    row['area'] = area

with open('output.csv', 'w') as csvfile:
    fieldnames = ['timestamp', 'x1', 'y1', 'x2', 'y2', 'area']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in data:
        writer.writerow(row)
