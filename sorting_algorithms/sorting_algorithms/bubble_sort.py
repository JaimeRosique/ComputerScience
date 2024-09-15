def bubble_sort(array: list[int]) -> list[int]:
    l = len(array)
    for i in range(l-1):
        for j in range(0, l-1):
            if (array[j] > array[j+1]):
                array[j], array[j+1] = array[j+1], array[j]
    return array
   
a = [5,4,2,3,1]     
print(bubble_sort(a)) 
#REPASAR