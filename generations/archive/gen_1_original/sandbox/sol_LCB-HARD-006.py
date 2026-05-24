def trapping_rain_water(heights):
    """
    Calculates the amount of trapped rain water given an elevation map.

    Args:
        heights: A list of non-negative integers representing the elevation map.

    Returns:
        The total amount of trapped rain water.
    """
    if not heights:
        return 0

    n = len(heights)
    left_max = [0] * n
    right_max = [0] * n
    water = 0

    left_max[0] = heights[0]
    for i in range(1, n):
        left_max[i] = max(heights[i], left_max[i - 1])

    right_max[n - 1] = heights[n - 1]
    for i in range(n - 2, -1, -1):
        right_max[i] = max(heights[i], right_max[i + 1])

    for i in range(n):
        water += min(left_max[i], right_max[i]) - heights[i]

    return water


# Test Cases
heights1 = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
expected1 = 6
result1 = trapping_rain_water(heights1)
print(f"Test Case 1: Heights = {heights1}, Expected = {expected1}, Result = {result1}")
assert result1 == expected1, f"Test Case 1 Failed: Expected {expected1}, got {result1}"

heights2 = [4, 2, 0, 3, 2, 5]
expected2 = 9
result2 = trapping_rain_water(heights2)
print(f"Test Case 2: Heights = {heights2}, Expected = {expected2}, Result = {result2}")
assert result2 == expected2, f"Test Case 2 Failed: Expected {expected2}, got {result2}"

heights3 = []
expected3 = 0
result3 = trapping_rain_water(heights3)
print(f"Test Case 3: Heights = {heights3}, Expected = {expected3}, Result = {result3}")
assert result3 == expected3, f"Test Case 3 Failed: Expected {expected3}, got {result3}"

heights4 = [2,1,0,3]
expected4 = 3
result4 = trapping_rain_water(heights4)
print(f"Test Case 4: Heights = {heights4}, Expected = {expected4}, Result = {result4}")
assert result4 == expected4, f"Test Case 4 Failed: Expected {expected4}, got {result4}"