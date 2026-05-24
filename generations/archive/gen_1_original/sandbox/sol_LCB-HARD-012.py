def maximal_rectangle(matrix):
    """
    Given a rows x cols binary matrix filled with 0's and 1's, find the largest rectangle
    containing only 1's and return its area.

    Args:
        matrix (List[List[str]]): A 2D list representing the binary matrix.

    Returns:
        int: The area of the largest rectangle containing only 1's.
    """

    if not matrix:
        return 0

    rows = len(matrix)
    cols = len(matrix[0])
    heights = [0] * cols
    max_area = 0

    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == '1':
                heights[j] += 1
            else:
                heights[j] = 0

        max_area = max(max_area, largest_rectangle_area(heights))

    return max_area


def largest_rectangle_area(heights):
    """
    Calculates the largest rectangular area possible in a given histogram represented
    by heights.

    Args:
        heights (List[int]): A list of integers representing the heights of the bars
                             in the histogram.

    Returns:
        int: The largest rectangular area.
    """

    stack = []
    max_area = 0
    i = 0
    while i < len(heights):
        if not stack or heights[i] > heights[stack[-1]]:
            stack.append(i)
            i += 1
        else:
            top = stack.pop()
            area = heights[top] * (i if not stack else i - stack[-1] - 1)
            max_area = max(max_area, area)

    while stack:
        top = stack.pop()
        area = heights[top] * (i if not stack else i - stack[-1] - 1)
        max_area = max(max_area, area)

    return max_area


# Test cases
matrix1 = [['1', '0', '1', '0', '0'], ['1', '0', '1', '1', '1'], ['1', '1', '1', '1', '1'], ['1', '0', '0', '1', '0']]
result1 = maximal_rectangle(matrix1)
print(f"Matrix 1: {matrix1}, Maximal Rectangle Area: {result1}")
assert result1 == 6

matrix2 = [["0", "1"], ["1", "0"]]
result2 = maximal_rectangle(matrix2)
print(f"Matrix 2: {matrix2}, Maximal Rectangle Area: {result2}")
assert result2 == 1

matrix3 = [["1", "1", "1"], ["1", "1", "1"], ["1", "1", "1"]]
result3 = maximal_rectangle(matrix3)
print(f"Matrix 3: {matrix3}, Maximal Rectangle Area: {result3}")
assert result3 == 9

matrix4 = []
result4 = maximal_rectangle(matrix4)
print(f"Matrix 4: {matrix4}, Maximal Rectangle Area: {result4}")
assert result4 == 0

matrix5 = [["0"]]
result5 = maximal_rectangle(matrix5)
print(f"Matrix 5: {matrix5}, Maximal Rectangle Area: {result5}")
assert result5 == 0

matrix6 = [["1"]]
result6 = maximal_rectangle(matrix6)
print(f"Matrix 6: {matrix6}, Maximal Rectangle Area: {result6}")
assert result6 == 1