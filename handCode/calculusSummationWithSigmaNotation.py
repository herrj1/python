#Methods definition
def sum(l):
    sum = 0
    for x in l:
        sum += 1/(pow(x,2)+9)
        sum += 1/(pow(x,2)+9)
    return sum

#Sample run using an arrays
l = [2,3,4,5,6]
sum(l)

#Sample output
0.22411261940673707
