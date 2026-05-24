```python
def max_non_adjacent_sum(nums):
    """
    Finds the maximum sum of a non-adjacent subsequence in a circular array.

    Args:
        nums: A list of integers representing house values.

    Returns:
        The maximum amount of money you can rob without alerting the police.
    """

    def rob_linear(arr):
        """
        Finds the maximum sum of a non-adjacent subsequence in a linear array.
        """
        if not arr:
            return 0
        if len(arr) <= 2:
            return max(arr) if arr else 0

        dp = [0] * len(arr)
        dp[0] = arr[0]
        dp[1] = max(arr[0], arr[1])

        for i in range(2, len(arr)):
            dp[i] = max(dp[i - 1], dp[i - 2] + arr[i])

        return dp[-1]

    if not nums:
        return 0
    if len(nums) <= 2:
        return max(nums) if nums else 0

    # Case 1: Include the first element, exclude the last
    case1 = rob_linear(nums[:-1])

    # Case 2: Exclude the first element, include the last
    case2 = rob_linear(nums[1:])

    return max(case1, case2)


# Test Cases
print(max_non_adjacent_sum([2, 3, 2]))
assert max_non_adjacent_sum([2, 3, 2]) == 3
print(max_non_adjacent_sum([1, 2, 3, 1]))
assert max_non_adjacent_sum([1, 2, 3, 1]) == 4
print(max_non_adjacent_sum([1, 2, 3]))
assert max_non_adjacent_sum([1, 2, 3]) == 4
print(max_non_adjacent_sum([1, 7, 9, 4]))
assert max_non_adjacent_sum([1, 7, 9, 4]) == 16
print(max_non_adjacent_sum([]))
assert max_non_adjacent_sum([]) == 0
print(max_non_adjacent_sum([5]))
assert max_non_adjacent_sum([5]) == 5
print(max_non_adjacent_sum([5, 2]))
assert max_non_adjacent_sum([5, 2]) == 5
print(max_non_adjacent_sum([5, 2, 8]))
assert max_non_adjacent_sum([5, 2, 8]) == 13
print(max_non_adjacent_sum([5, 2, 8, 1]))
assert max_non_adjacent_sum([5, 2, 8, 1]) == 13
print(max_non_adjacent_sum([5, 2, 8, 1, 9]))
assert max_non_adjacent_sum([5, 2, 8, 1, 9]) == 22
```