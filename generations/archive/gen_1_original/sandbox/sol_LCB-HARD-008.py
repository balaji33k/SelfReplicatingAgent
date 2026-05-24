def solve_n_queens(n):
    def is_safe(board, row, col):
        # Check same column
        for i in range(row):
            if board[i][col] == 'Q':
                return False

        # Check upper left diagonal
        for i, j in zip(range(row - 1, -1, -1), range(col - 1, -1, -1)):
            if board[i][j] == 'Q':
                return False

        # Check upper right diagonal
        for i, j in zip(range(row - 1, -1, -1), range(col + 1, n)):
            if board[i][j] == 'Q':
                return False

        return True

    def solve_n_queens_util(board, row, solutions):
        if row == n:
            solution = ["".join(row) for row in board]
            solutions.append(solution)
            return

        for col in range(n):
            if is_safe(board, row, col):
                board[row][col] = 'Q'
                solve_n_queens_util(board, row + 1, solutions)
                board[row][col] = '.'  # Backtrack

    board = [['.' for _ in range(n)] for _ in range(n)]
    solutions = []
    solve_n_queens_util(board, 0, solutions)
    return solutions


# Test Cases
def test_solve_n_queens():
    expected_4 = [['.Q..', '...Q', 'Q...', '..Q.'], ['..Q.', 'Q...', '...Q', '.Q..']]
    actual_4 = solve_n_queens(4)

    # Sort the inner lists so that the order does not affect the verification
    for solution in actual_4:
        solution.sort()
    for solution in expected_4:
        solution.sort()
    actual_4.sort()
    expected_4.sort()

    assert actual_4 == expected_4, f"Failed for n=4, expected {expected_4}, got {actual_4}"

    print("All test cases passed!")

test_solve_n_queens()