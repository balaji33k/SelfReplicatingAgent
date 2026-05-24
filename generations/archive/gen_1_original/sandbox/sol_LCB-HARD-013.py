class TrieNode:
    def __init__(self):
        self.children = {}
        self.word = None

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.word = word

class Solution:
    def findWords(self, board, words):
        trie = Trie()
        for word in words:
            trie.insert(word)

        rows, cols = len(board), len(board[0])
        result = set()
        visited = set()

        def dfs(row, col, node):
            if (row, col) in visited or \
               row < 0 or row >= rows or \
               col < 0 or col >= cols or \
               board[row][col] not in node.children:
                return

            visited.add((row, col))
            char = board[row][col]
            node = node.children[char]

            if node.word:
                result.add(node.word)

            dfs(row + 1, col, node)
            dfs(row - 1, col, node)
            dfs(row, col + 1, node)
            dfs(row, col - 1, node)

            visited.remove((row, col))  # Backtrack

        for row in range(rows):
            for col in range(cols):
                dfs(row, col, trie.root)

        return list(result)


# Test cases
board1 = [['o', 'a', 'a', 'n'], ['e', 't', 'a', 'e'], ['i', 'h', 'k', 'r'], ['i', 'f', 'l', 'v']]
words1 = ['oath', 'pea', 'eat', 'rain']
expected1 = ['eat', 'oath']
result1 = Solution().findWords(board1, words1)
assert sorted(result1) == sorted(expected1), f"Test Case 1 Failed: Expected {expected1}, Got {result1}"

board2 = [["a","b"],["c","d"]]
words2 = ["abcb"]
expected2 = []
result2 = Solution().findWords(board2, words2)
assert sorted(result2) == sorted(expected2), f"Test Case 2 Failed: Expected {expected2}, Got {result2}"

board3 = [["a"]]
words3 = ["a"]
expected3 = ["a"]
result3 = Solution().findWords(board3, words3)
assert sorted(result3) == sorted(expected3), f"Test Case 3 Failed: Expected {expected3}, Got {result3}"

board4 = [["a", "b"], ["c", "a"]]
words4 = ["acb"]
expected4 = []
result4 = Solution().findWords(board4, words4)
assert sorted(result4) == sorted(expected4), f"Test Case 4 Failed: Expected {expected4}, Got {result4}"

print("All test cases passed!")