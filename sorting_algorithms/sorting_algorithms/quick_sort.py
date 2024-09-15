def quick_sort(array:list[int], low:int, high:int) -> list[int]:
    """
    Sorts a list of number in ascending order using the algorithm quick sort.
    
    Quick sort picks an element as a pivot and partitions the given array around the picked pivot by placing the pivot in its correct position in the sorted array,
    having numbers less than pivot on the left and the greater numbers on the right. 

    Args:
        array (list[int]): Unsorted list of intgers to be sorted
        low (int): The starting index of the portion of list to be sorted
        high (int): The ending index of the list to be sorted. 
    Returns:
        int: The sorted list of integers in ascending order.
    """    
    if (low < high):
        pivot: int = makePivot(array, low, high)
        quick_sort(array, low, pivot)
        quick_sort(array, pivot+1, high)
    return array

def makePivot(array:list[int], low:int, high:int) -> int:
    """
    Rearranges the array so that all elements smaller than the pivot are on the left, and all elements larger are on the right.

    Args:
        array (list[int]): The array to partition around the pivot.
        low (int): The lowest index of the range to consider for partitioning.
        high (int): The highest index of the range to consider for partitioning.

    Returns:
        int: The index where the pivot is placed after partitioning.
    """

    pivot = array[low]
    pivot_index = low
    
    for i in range(low+1, high):
        if (array[i] < pivot):
            array[i], array[pivot_index] = array[pivot_index], array[i]
            pivot_index = pivot_index + 1
    array[array.index(pivot)], array[pivot_index] = array[pivot_index], array[array.index(pivot)]
    
    return pivot_index
