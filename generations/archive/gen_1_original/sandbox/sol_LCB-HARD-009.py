class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverseKGroup(head, k):
    """
    Reverses the nodes of a linked list k at a time.

    Args:
        head: The head of the linked list.
        k: The number of nodes to reverse at a time.

    Returns:
        The head of the modified linked list.
    """

    if not head or k <= 1:
        return head

    dummy = ListNode(0)
    dummy.next = head
    prev = dummy
    curr = head

    while True:
        count = 0
        temp = curr
        while temp and count < k:
            temp = temp.next
            count += 1

        if count < k:
            break

        # Reverse the k nodes
        next_node = curr.next
        tail = curr
        for _ in range(k - 1):
            next_node_next = next_node.next
            next_node.next = curr
            curr = next_node
            next_node = next_node_next

        # Update pointers
        prev.next = curr
        tail.next = next_node

        # Move pointers to the next group
        prev = tail
        curr = next_node

    return dummy.next

def list_to_linked_list(lst):
    """Converts a list to a linked list."""
    if not lst:
        return None
    head = ListNode(lst[0])
    curr = head
    for i in range(1, len(lst)):
        curr.next = ListNode(lst[i])
        curr = curr.next
    return head

def linked_list_to_list(head):
    """Converts a linked list to a list."""
    lst = []
    curr = head
    while curr:
        lst.append(curr.val)
        curr = curr.next
    return lst

# Test cases
# Test case 1
head1 = list_to_linked_list([1, 2, 3, 4, 5])
k1 = 2
reversed_head1 = reverseKGroup(head1, k1)
result1 = linked_list_to_list(reversed_head1)
expected1 = [2, 1, 4, 3, 5]
assert result1 == expected1, f"Test Case 1 Failed: Expected {expected1}, got {result1}"

# Test case 2
head2 = list_to_linked_list([1, 2, 3, 4, 5])
k2 = 3
reversed_head2 = reverseKGroup(head2, k2)
result2 = linked_list_to_list(reversed_head2)
expected2 = [3, 2, 1, 4, 5]
assert result2 == expected2, f"Test Case 2 Failed: Expected {expected2}, got {result2}"

# Test case 3: k = 1
head3 = list_to_linked_list([1, 2, 3, 4, 5])
k3 = 1
reversed_head3 = reverseKGroup(head3, k3)
result3 = linked_list_to_list(reversed_head3)
expected3 = [1, 2, 3, 4, 5]
assert result3 == expected3, f"Test Case 3 Failed: Expected {expected3}, got {result3}"

# Test case 4: k > length of list
head4 = list_to_linked_list([1, 2, 3])
k4 = 4
reversed_head4 = reverseKGroup(head4, k4)
result4 = linked_list_to_list(reversed_head4)
expected4 = [1, 2, 3]
assert result4 == expected4, f"Test Case 4 Failed: Expected {expected4}, got {result4}"

# Test case 5: Empty list
head5 = list_to_linked_list([])
k5 = 2
reversed_head5 = reverseKGroup(head5, k5)
result5 = linked_list_to_list(reversed_head5)
expected5 = []
assert result5 == expected5, f"Test Case 5 Failed: Expected {expected5}, got {result5}"

print("All test cases passed!")