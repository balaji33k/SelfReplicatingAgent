def find_median_sorted_arrays(nums1, nums2):
    """
    Finds the median of two sorted arrays.

    Args:
        nums1: The first sorted array.
        nums2: The second sorted array.

    Returns:
        The median of the two sorted arrays.
    """
    m, n = len(nums1), len(nums2)
    if m > n:
        nums1, nums2 = nums2, nums1
        m, n = n, m

    low, high = 0, m
    while low <= high:
        partitionX = (low + high) // 2
        partitionY = (m + n + 1) // 2 - partitionX

        maxLeftX = nums1[partitionX - 1] if partitionX > 0 else float('-inf')
        minRightX = nums1[partitionX] if partitionX < m else float('inf')

        maxLeftY = nums2[partitionY - 1] if partitionY > 0 else float('-inf')
        minRightY = nums2[partitionY] if partitionY < n else float('inf')

        if maxLeftX <= minRightY and maxLeftY <= minRightX:
            if (m + n) % 2 == 0:
                return (max(maxLeftX, maxLeftY) + min(minRightX, minRightY)) / 2
            else:
                return max(maxLeftX, maxLeftY)
        elif maxLeftX > minRightY:
            high = partitionX - 1
        else:
            low = partitionX + 1


# Test Cases
def test_find_median_sorted_arrays():
    # Test Case 1: [1,3], [2] -> 2.0
    nums1_1, nums2_1 = [1, 3], [2]
    expected_1 = 2.0
    actual_1 = find_median_sorted_arrays(nums1_1, nums2_1)
    assert abs(actual_1 - expected_1) < 1e-5, f"Test Case 1 Failed: Expected {expected_1}, but got {actual_1}"
    print("Test Case 1 Passed")

    # Test Case 2: [1,2], [3,4] -> 2.5
    nums1_2, nums2_2 = [1, 2], [3, 4]
    expected_2 = 2.5
    actual_2 = find_median_sorted_arrays(nums1_2, nums2_2)
    assert abs(actual_2 - expected_2) < 1e-5, f"Test Case 2 Failed: Expected {expected_2}, but got {actual_2}"
    print("Test Case 2 Passed")

    # Test Case 3: [0,0], [0,0] -> 0.0
    nums1_3, nums2_3 = [0,0], [0,0]
    expected_3 = 0.0
    actual_3 = find_median_sorted_arrays(nums1_3, nums2_3)
    assert abs(actual_3 - expected_3) < 1e-5, f"Test Case 3 Failed: Expected {expected_3}, but got {actual_3}"
    print("Test Case 3 Passed")

    # Test Case 4: [], [1] -> 1.0
    nums1_4, nums2_4 = [], [1]
    expected_4 = 1.0
    actual_4 = find_median_sorted_arrays(nums1_4, nums2_4)
    assert abs(actual_4 - expected_4) < 1e-5, f"Test Case 4 Failed: Expected {expected_4}, but got {actual_4}"
    print("Test Case 4 Passed")

    # Test Case 5: [2], [] -> 2.0
    nums1_5, nums2_5 = [2], []
    expected_5 = 2.0
    actual_5 = find_median_sorted_arrays(nums1_5, nums2_5)
    assert abs(actual_5 - expected_5) < 1e-5, f"Test Case 5 Failed: Expected {expected_5}, but got {actual_5}"
    print("Test Case 5 Passed")

    # Test Case 6: [1, 2, 5], [3, 4] -> 3.0
    nums1_6, nums2_6 = [1, 2, 5], [3, 4]
    expected_6 = 3.0
    actual_6 = find_median_sorted_arrays(nums1_6, nums2_6)
    assert abs(actual_6 - expected_6) < 1e-5, f"Test Case 6 Failed: Expected {expected_6}, but got {actual_6}"
    print("Test Case 6 Passed")

test_find_median_sorted_arrays()