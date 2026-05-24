def longest_valid_parentheses(s):
    """
    Given a string containing just the characters '(' and ')', find the length of the longest valid (well-formed) parentheses substring.

    Args:
        s (str): The input string containing parentheses.

    Returns:
        int: The length of the longest valid parentheses substring.
    """

    max_len = 0
    stack = [-1]  # Initialize the stack with -1 to handle cases where the entire string is valid.

    for i, char in enumerate(s):
        if char == '(':
            stack.append(i)
        else:
            stack.pop()
            if not stack:
                stack.append(i)
            else:
                max_len = max(max_len, i - stack[-1])

    return max_len


# Test cases
def test_longest_valid_parentheses():
    assert longest_valid_parentheses("(()") == 2, "Test Case 1 Failed: (()"
    assert longest_valid_parentheses(")()())") == 4, "Test Case 2 Failed: )()())"
    assert longest_valid_parentheses("") == 0, "Test Case 3 Failed: Empty string"
    assert longest_valid_parentheses("()") == 2, "Test Case 4 Failed: ()"
    assert longest_valid_parentheses("(()())") == 6, "Test Case 5 Failed: (()())"
    assert longest_valid_parentheses("))((())") == 4, "Test Case 6 Failed: ))((())"
    assert longest_valid_parentheses("()(())") == 6, "Test Case 7 Failed: ()(())"
    assert longest_valid_parentheses("(()(((()") == 2, "Test Case 8 Failed: (()(((()"
    assert longest_valid_parentheses(")") == 0, "Test Case 9 Failed: )"
    assert longest_valid_parentheses("((()))())()") == 8, "Test Case 10 Failed: ((()))())()"

    print("All test cases passed!")

if __name__ == "__main__":
    test_longest_valid_parentheses()