import csv

l = []

# generate 5 digit numbers
i = 10001
while True:
    if len(l) == 4096:
        break
    i += 1
    if i % 10 == 0:
        continue

    l.append([*list(str(i) + str(i)[::-1]), 1])
    l.append([*list(str(i) + str(i+1)), 0])

with open('data.csv', 'w+', newline ='') as file:
    write = csv.writer(file)
    write.writerows(l)