class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def maxPathSum(root):
    """
    Finds the maximum path sum of any non-empty path in a binary tree.

    Args:
        root: The root of the binary tree.

    Returns:
        The maximum path sum.
    """

    max_sum = float('-inf')

    def max_gain(node):
        """
        Recursively calculates the maximum gain from a node (including the node itself)
        to either its left or right subtree, or just the node itself.

        Args:
            node: The current node.

        Returns:
            The maximum gain from the node.
        """
        nonlocal max_sum

        if not node:
            return 0

        # max sum from left and right subtrees, if negative consider them to be zero
        left_gain = max(max_gain(node.left), 0)
        right_gain = max(max_gain(node.right), 0)

        # the price to start a new path where `node` is a highest node
        price_newpath = node.val + left_gain + right_gain

        # update max_sum if it's better to start a new path
        max_sum = max(max_sum, price_newpath)

        # for recursion :
        # return the max gain if continue the same path
        return node.val + max(left_gain, right_gain)

    max_gain(root)
    return max_sum


# Test Cases
# Create tree from list representation
def create_tree(arr):
    if not arr:
        return None

    nodes = [TreeNode(val) if val is not None else None for val in arr]
    
    for i in range(len(nodes)):
        if nodes[i] is not None:
            left_index = 2 * i + 1
            right_index = 2 * i + 2
            
            if left_index < len(nodes):
                nodes[i].left = nodes[left_index]
            if right_index < len(nodes):
                nodes[i].right = nodes[right_index]
                
    return nodes[0]


# Test Case 1: [1,2,3] -> 6
root1 = create_tree([1, 2, 3])
result1 = maxPathSum(root1)
print(f"Test Case 1: {result1}")
assert result1 == 6, f"Test Case 1 Failed: Expected 6, got {result1}"

# Test Case 2: [-10,9,20,null,null,15,7] -> 42
root2 = create_tree([-10, 9, 20, None, None, 15, 7])
result2 = maxPathSum(root2)
print(f"Test Case 2: {result2}")
assert result2 == 42, f"Test Case 2 Failed: Expected 42, got {result2}"

# Additional Test Cases
root3 = create_tree([2, -1, -2])
result3 = maxPathSum(root3)
print(f"Test Case 3: {result3}")
assert result3 == 2, f"Test Case 3 Failed: Expected 2, got {result3}"

root4 = create_tree([-3])
result4 = maxPathSum(root4)
print(f"Test Case 4: {result4}")
assert result4 == -3, f"Test Case 4 Failed: Expected -3, got {result4}"

root5 = create_tree([5,4,8,11,None,13,4,7,2,None,None,None,1])
result5 = maxPathSum(root5)
print(f"Test Case 5: {result5}")
assert result5 == 48, f"Test Case 5 Failed: Expected 48, got {result5}"

print("All Test Cases Passed!")