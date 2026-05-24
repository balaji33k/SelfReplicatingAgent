def house_robber(nums):
    """
    Finds the maximum sum of a non-adjacent subsequence in a circular array.

    Args:
        nums: A list of integers representing the amount of money in each house.

    Returns:
        The maximum amount of money that can be robbed.
    """

    def rob(nums):
        if not nums:
            return 0
        if len(nums) <= 2:
            return max(nums) if nums else 0

        dp = [0] * len(nums)
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])

        for i in range(2, len(nums)):
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])

        return dp[-1]

    if not nums:
        return 0
    if len(nums) <= 2:
        return max(nums) if nums else 0
    
    # Option 1: Rob the first house, so we can't rob the last house
    option1 = rob(nums[:-1])
    
    # Option 2: Don't rob the first house, so we can rob the last house
    option2 = rob(nums[1:])
    
    return max(option1, option2)

# Test cases
def test_house_robber():
    assert house_robber([2, 3, 2]) == 3
    assert house_robber([1, 2, 3, 1]) == 4
    assert house_robber([1, 2, 3]) == 3
    assert house_robber([1]) == 1
    assert house_robber([]) == 0
    assert house_robber([1, 2]) == 2
    assert house_robber([5, 5, 10, 100, 10, 5]) == 110
    assert house_robber([4, 1, 2, 7, 5, 3, 1]) == 14

test_house_robber()