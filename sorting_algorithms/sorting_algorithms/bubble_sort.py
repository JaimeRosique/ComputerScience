def bubble_sort(array: list[int]) -> list[int]:
    """
    Sorts a list of number in ascending order using the algorithm bubble sort.
    
    Bubble sort works by examining each set of adjacent elements in the list, from left to right, switching their positions if they are out of order

    Args:
        array (list[int]): Unsorted list of intgers to be sorted

    Returns:
        list[int]: Sorted list of integers in ascending order
    """    
    l = len(array)
    for i in range(l-1):
        for j in range(0, l-1):
            if (array[j] > array[j+1]):
                array[j], array[j+1] = array[j+1], array[j]
    return array
   