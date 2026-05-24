import re

def is_match(s: str, p: str) -> bool:
    """
    Implement regular expression matching with support for '.' (matching any single character)
    and '*' (matching zero or more of the preceding element).

    Args:
        s: The input string.
        p: The pattern.

    Returns:
        True if the pattern matches the string, False otherwise.
    """
    return bool(re.fullmatch(p, s))

# Test Cases
def run_tests():
    assert is_match("aa", "a*") == True, "Test Case 1 Failed"
    assert is_match("ab", ".*") == True, "Test Case 2 Failed"
    assert is_match("aab", "c*a*b") == True, "Test Case 3 Failed"
    assert is_match("mississippi", "mis*is*p*.") == False, "Test Case 4 Failed"
    assert is_match("aaa", "a*a") == True, "Test Case 5 Failed"
    assert is_match("aaa", "ab*a") == False, "Test Case 6 Failed"
    assert is_match("a", "ab*") == True, "Test Case 7 Failed"
    assert is_match("abc", ".*c") == True, "Test Case 8 Failed"
    assert is_match("bbcacbabbbbbaa", "b*a*a*.c*bb*b*.*.*") == True, "Test Case 9 Failed"
    assert is_match("aaaaaa", "a*") == True, "Test Case 10 Failed"
    assert is_match("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab", "a*b") == True, "Test Case 11 Failed"

    print("All test cases passed!")

if __name__ == "__main__":
    run_tests()