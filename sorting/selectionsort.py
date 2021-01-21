def selectionsort(arr):
    currentIndex = 0
    while(currentIndex < len(arr) - 1):
        minIndex = currentIndex
        
        i = currentIndex +1
        while(i < len(arr)):
            if(arr[i] < arr[minIndex]):
                minIndex = i
            i +=1
        
        if(minIndex != currentIndex):
            temp = arr[currentIndex]
            arr[currentIndex] = arr[minIndex]
            arr[minIndex] = temp
        currentIndex +=1

if __name__ == '__main__'
    arr = [20,1,34,2,7,3]
    selectionsort(arr)
    print(arr)