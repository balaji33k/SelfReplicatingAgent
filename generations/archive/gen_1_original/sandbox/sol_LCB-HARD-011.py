def edit_distance(word1, word2):
    """
    Calculates the minimum edit distance between two strings.

    Args:
        word1 (str): The first string.
        word2 (str): The second string.

    Returns:
        int: The minimum edit distance.
    """
    n = len(word1)
    m = len(word2)

    # Create a DP table to store the minimum edit distances
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # Initialize the first row and column
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    # Populate the DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],      # Deletion
                                   dp[i][j - 1],      # Insertion
                                   dp[i - 1][j - 1])  # Replacement

    return dp[n][m]

# Test cases
def run_tests():
    # Test case 1
    word1 = "horse"
    word2 = "ros"
    expected_distance = 3
    actual_distance = edit_distance(word1, word2)
    assert actual_distance == expected_distance, f"Test Case 1 Failed: Expected {expected_distance}, got {actual_distance}"
    print("Test Case 1 Passed")

    # Test case 2
    word1 = "intention"
    word2 = "execution"
    expected_distance = 5
    actual_distance = edit_distance(word1, word2)
    assert actual_distance == expected_distance, f"Test Case 2 Failed: Expected {expected_distance}, got {actual_distance}"
    print("Test Case 2 Passed")
    
    # Test case 3
    word1 = ""
    word2 = "abc"
    expected_distance = 3
    actual_distance = edit_distance(word1, word2)
    assert actual_distance == expected_distance, f"Test Case 3 Failed: Expected {expected_distance}, got {actual_distance}"
    print("Test Case 3 Passed")

    # Test case 4
    word1 = "abc"
    word2 = ""
    expected_distance = 3
    actual_distance = edit_distance(word1, word2)
    assert actual_distance == expected_distance, f"Test Case 4 Failed: Expected {expected_distance}, got {actual_distance}"
    print("Test Case 4 Passed")
    
    # Test case 5
    word1 = "abc"
    word2 = "abc"
    expected_distance = 0
    actual_distance = edit_distance(word1, word2)
    assert actual_distance == expected_distance, f"Test Case 5 Failed: Expected {expected_distance}, got {actual_distance}"
    print("Test Case 5 Passed")
    

run_tests()