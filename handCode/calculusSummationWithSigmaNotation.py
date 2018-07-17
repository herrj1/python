#Method definition
def sum(l):
    sum = 0
    for x in l:
        sum += 1/(pow(x,2)+9)
    return sum

#Sample run
l = [2,3,4,5,6]
sum(l)

#Output
0.22411261940673707
