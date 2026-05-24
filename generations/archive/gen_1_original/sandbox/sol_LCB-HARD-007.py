def is_match(s: str, p: str) -> bool:
    """
    Given an input string (s) and a pattern (p), implement wildcard pattern matching with support for:

    '?' Matches any single character.
    '*' Matches any sequence of characters (including the empty sequence).

    The matching should cover the entire input string (not partial).

    Args:
        s: The input string.
        p: The pattern string.

    Returns:
        True if the pattern matches the string, False otherwise.
    """

    s_len = len(s)
    p_len = len(p)

    dp = [[False] * (p_len + 1) for _ in range(s_len + 1)]

    dp[0][0] = True

    for j in range(1, p_len + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 1]

    for i in range(1, s_len + 1):
        for j in range(1, p_len + 1):
            if p[j - 1] == '?':
                dp[i][j] = dp[i - 1][j - 1]
            elif p[j - 1] == '*':
                dp[i][j] = dp[i - 1][j] or dp[i][j - 1]
            else:
                dp[i][j] = dp[i - 1][j - 1] and (s[i - 1] == p[j - 1])

    return dp[s_len][p_len]


# Test cases
print(f"is_match('aa', '*') = {is_match('aa', '*')}")
assert is_match('aa', '*') == True

print(f"is_match('cb', '?a') = {is_match('cb', '?a')}")
assert is_match('cb', '?a') == False

print(f"is_match('adceb', '*a*b') = {is_match('adceb', '*a*b')}")
assert is_match('adceb', '*a*b') == True

print(f"is_match('acdcb', 'a*c?b') = {is_match('acdcb', 'a*c?b')}")
assert is_match('acdcb', 'a*c?b') == False

print(f"is_match('', '*') = {is_match('', '*')}")
assert is_match('', '*') == True

print(f"is_match('', 'a') = {is_match('', 'a')}")
assert is_match('', 'a') == False

print(f"is_match('a', '') = {is_match('a', '')}")
assert is_match('a', '') == False

print(f"is_match('a', 'a') = {is_match('a', 'a')}")
assert is_match('a', 'a') == True

print(f"is_match('ab', 'a?') = {is_match('ab', 'a?')}")
assert is_match('ab', 'a?') == True

print(f"is_match('abc', 'a*c') = {is_match('abc', 'a*c')}")
assert is_match('abc', 'a*c') == True

print(f"is_match('abc', 'a*b') = {is_match('abc', 'a*b')}")
assert is_match('abc', 'a*b') == False

print(f"is_match('abbbbbc', 'a*c') = {is_match('abbbbbc', 'a*c')}")
assert is_match('abbbbbc', 'a*c') == True

print(f"is_match('aaaaaaaaaaab', 'a*b') = {is_match('aaaaaaaaaaab', 'a*b')}")
assert is_match('aaaaaaaaaaab', 'a*b') == True

print(f"is_match('ho', 'ho**') = {is_match('ho', 'ho**')}")
assert is_match('ho', 'ho**') == True