def first_missing_positive(nums):
    """
    Finds the smallest missing positive integer in an unsorted array.

    Args:
        nums: A list of integers.

    Returns:
        The smallest missing positive integer.
    """
    n = len(nums)

    # Base case.
    if 1 not in nums:
        return 1

    # Replace negative numbers, zeros,
    # and numbers larger than n by 1s.
    # After this conversion, nums will contain
    # only positive numbers.
    for i in range(n):
        if nums[i] <= 0 or nums[i] > n:
            nums[i] = 1

    # Use the index as a hash key and number sign as a presence detector.
    # For example, if nums[1] is negative, that means that the number `1`
    # is present in the array.
    # If nums[2] is positive, the number 2 is missing.
    for i in range(n):
        a = abs(nums[i])
        # If you meet number a in the array, change the sign of the a-th element.
        # Be careful with duplicates: do it only once.
        if a == n:
            nums[0] = - abs(nums[0])
        else:
            nums[a] = - abs(nums[a])

    # Now the index of the first positive number
    # is equal to the first missing positive.
    for i in range(1, n):
        if nums[i] > 0:
            return i

    if nums[0] > 0:
        return n

    return n + 1


# Test cases
test_cases = [
    ([1, 2, 0], 3),
    ([3, 4, -1, 1], 2),
    ([7, 8, 9, 11, 12], 1),
    ([1, 2, 3], 4),
    ([-1, -2, -3], 1),
    ([1], 2),
    ([2], 1),
    ([2,2], 1),
    ([1, 1], 2),
    ([0, 2, 2, 1, 1], 3),
    ([1, 2, 3, 10, 21, 11], 4),
    ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 21)

]

for nums, expected in test_cases:
    result = first_missing_positive(nums)
    print(f"Input: {nums}, Expected: {expected}, Result: {result}")
    assert result == expected, f"Test failed for input {nums}. Expected {expected}, got {result}"

print("All test cases passed!")