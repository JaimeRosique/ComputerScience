def merge_sort(array: list[int]) -> list[int]:
    """
    Uses the merge sort algorithm to divide the array into individual elements and merge them into a single sorted array.

    Args:
        array (list[int]): Array that is used to sort the elements in it.

    Returns:
        list[int]: An array of elements sorted using the merge function
    """    
    l = len(array)
    if l == 1:
        return array
    half = l//2
    left = merge_sort(array[:half])
    right = merge_sort(array[half:])
    
    return merge(left, right)
    
def merge(left: list[int], right: list[int]) -> list[int]:
    """
    Merges two sorted lists into one sorted list.

    Args:
        left (list[int]): First list of elements to be merged into the sorted list
        right (list[int]): Second list of elements to be merged into the sorted list

    Returns:
        list[int]: A sorted list containing all the elements from the left list and from the right list
    """    
    merged = []
    i = j = 0
    while i < len(left) and j < len(right):
        if (left[i] > right[j]):
            merged.append(right[j])
            j += 1
        else:
            merged.append(left[i])
            i += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    
    return merged
    
print(merge_sort([3,4,5,2,1,6]))