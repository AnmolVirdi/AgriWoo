import categories
import csv
for i in categories.category.values():
    for j in i[1].values():
        if j==0:
            # print("yea")
            with open('disease.csv', 'r') as f:
                mycsv = list(csv.reader(f))
                print(type(mycsv[1][1]))
            break