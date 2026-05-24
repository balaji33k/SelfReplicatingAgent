class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def recoverTree(root):
    """
    Recovers a binary search tree (BST) where two nodes have been swapped.

    Args:
        root: The root of the BST.

    Returns:
        None (modifies the tree in-place).
    """

    first = None
    second = None
    prev = None

    def inorder(node):
        nonlocal first, second, prev

        if not node:
            return

        inorder(node.left)

        if prev and node.val < prev.val:
            if not first:
                first = prev
                second = node
            else:
                second = node
        prev = node

        inorder(node.right)

    inorder(root)

    if first and second:
        first.val, second.val = second.val, first.val
def build_tree(arr):
    """Builds a binary tree from a list representation."""
    if not arr:
        return None

    nodes = [TreeNode(x) if x is not None else None for x in arr]
    
    root = nodes[0] if nodes else None
    
    for i, node in enumerate(nodes):
        if node:
            left_index = 2 * i + 1
            right_index = 2 * i + 2
            
            if left_index < len(nodes):
                node.left = nodes[left_index]
            if right_index < len(nodes):
                node.right = nodes[right_index]
    return root

def tree_to_list(root):
    """Converts a binary tree to a list representation."""
    if not root:
        return []

    result = []
    queue = [root]

    while queue:
        node = queue.pop(0)
        if node:
            result.append(node.val)
            queue.append(node.left)
            queue.append(node.right)
        else:
            result.append(None)

    # Remove trailing None values
    while result and result[-1] is None:
        result.pop()

    return result

# Test case 1: [1,3,null,null,2] -> [3,1,null,null,2]
arr1 = [1, 3, None, None, 2]
root1 = build_tree(arr1)
recoverTree(root1)
result1 = tree_to_list(root1)
expected1 = [3, 1, None, None, 2]
print(f"Test Case 1: Input: {arr1}, Output: {result1}, Expected: {expected1}")
assert result1 == expected1

# Test case 2: [3,1,4,None,None,2] -> [2,1,4,None,None,3]
arr2 = [3, 1, 4, None, None, 2]
root2 = build_tree(arr2)
recoverTree(root2)
result2 = tree_to_list(root2)
expected2 = [2, 1, 4, None, None, 3]
print(f"Test Case 2: Input: {arr2}, Output: {result2}, Expected: {expected2}")
assert result2 == expected2

# Test case 3: Empty tree
arr3 = []
root3 = build_tree(arr3)
recoverTree(root3)
result3 = tree_to_list(root3)
expected3 = []
print(f"Test Case 3: Input: {arr3}, Output: {result3}, Expected: {expected3}")
assert result3 == expected3

# Test case 4: Single node tree
arr4 = [1]
root4 = build_tree(arr4)
recoverTree(root4)
result4 = tree_to_list(root4)
expected4 = [1]
print(f"Test Case 4: Input: {arr4}, Output: {result4}, Expected: {expected4}")
assert result4 == expected4

# Test case 5: Almost sorted [10,5,15,None,None,6,20] -> [10,6,15,None,None,5,20]
arr5 = [10,5,15,None,None,6,20]
root5 = build_tree(arr5)
recoverTree(root5)
result5 = tree_to_list(root5)
expected5 = [10, 6, 15, None, None, 5, 20]

print(f"Test Case 5: Input: {arr5}, Output: {result5}, Expected: {expected5}")
assert result5 == expected5