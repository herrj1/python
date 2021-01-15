def pivot(array, start, end):
    #initialize
    pivot = array[start]
    low = start + 1
    high = end
    
    while True:
        while low <= high and array[high] >= pivot:
            high = high -1
            
        while low <= high and array[low] <= pivot:
            low = low + 1
            
        if low <= high:
            array[low], array[high] = array[high], array[low]
        else:
            break
            
    array[start], array[high] = array[high], array[start]
    
    return high
    
def quicksort (array, start, end):
    if start >= end:
        return 
        
    p = pivot(array, start, end)
    quicksort(array,start,p-1)
    quicksort(array,p+1, end)
    
array = [6,3,3,2,9,2]
quicksort(array, 0, len(array)-1)
print(array)