```python
import sys

class SWETrain0Hard:
    """
    Logic for SWE-TRAIN-0 (Difficulty: Hard).
    Implements a recursive backtracking engine with memoization to solve 
    complex pattern matching (Regular Expression Matching) containing '.' and '*'.
    """
    def __init__(self):
        self.memo = {}

    def is_match(self, s: str, p: str) -> bool:
        self.memo = {}
        return self._compute(0, 0, s, p)

    def _compute(self, i: int, j: int, s: str, p: str) -> bool:
        state = (i, j)
        if state in self.memo:
            return self.memo[state]

        # If we reached the end of the pattern, check if we're at the end of the string
        if j == len(p):
            return i == len(s)

        # Check if the current characters match
        first_match = i < len(s) and (p[j] == s[i] or p[j] == '.')

        # Handle the '*' wildcard logic
        if j + 1 < len(p) and p[j+1] == '*':
            # Two possibilities:
            # 1. Ignore the '*' and its preceding element (move pattern index by 2)
            # 2. Consume one matching character from string and stay on the same pattern index
            ans = self._compute(i, j + 2, s, p) or (first_match and self._compute(i + 1, j, s, p))
        else:
            # Normal character match logic
            ans = first_match and self._compute(i + 1, j + 1, s, p)

        self.memo[state] = ans
        return ans

def run_verification():
    """
    Test Suite to verify the logic against Hard difficulty requirements.
    """
    solver = SWETrain0Hard()

    test_cases = [
        # (input_string, pattern, expected_outcome)
        ("aa", "a", False),
        ("aa", "a*", True),
        ("ab", ".*", True),
        ("aab", "c*a*b", True),
        ("mississippi", "mis*is*p*.", False),
        ("", ".*", True),
        ("a", "ab*", True),
        ("bbbba", ".*a*a", True),
        ("ab", ".*c", False),
        ("aaa", "a*a", True),
        ("aaa", "ab*a*c*a", True),
        ("", "", True),
        ("abcde", ".*e", True),
        ("a", ".*..a*", False)
    ]

    for s, p, expected in test_cases:
        result = solver.is_match(s, p)
        assert result == expected, f"Assertion Failed: s='{s}', p='{p}' | Expected: {expected}, Got: {result}"

    print("SUCCESS: SWE-TRAIN-0 Hard verification passed all test cases.")

if __name__ == "__main__":
    run_verification()
```