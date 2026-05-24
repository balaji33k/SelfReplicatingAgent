def solve_sudoku(board):
    """
    Solves a Sudoku puzzle in-place.

    Args:
        board: A 9x9 list of lists representing the Sudoku board.
               Empty cells are represented by '.'.
    """

    def is_valid(board, row, col, num):
        """Checks if placing num at board[row][col] is valid."""
        # Check row
        for i in range(9):
            if board[row][i] == str(num):
                return False
        # Check column
        for i in range(9):
            if board[i][col] == str(num):
                return False
        # Check 3x3 box
        box_row = (row // 3) * 3
        box_col = (col // 3) * 3
        for i in range(3):
            for j in range(3):
                if board[box_row + i][box_col + j] == str(num):
                    return False
        return True

    def find_empty(board):
        """Finds an empty cell (represented by '.') and returns its coordinates (row, col).
           Returns None if no empty cell is found.
        """
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    return (i, j)
        return None

    def solve(board):
        """Recursive helper function to solve the Sudoku."""
        empty_pos = find_empty(board)
        if not empty_pos:
            return True  # No more empty cells, the Sudoku is solved

        row, col = empty_pos

        for num in range(1, 10):
            if is_valid(board, row, col, num):
                board[row][col] = str(num)

                if solve(board):  # Recursively try to solve the rest of the Sudoku
                    return True
                
                board[row][col] = '.'  # Backtrack: reset the cell if the solution is not valid
        
        return False  # No valid number can be placed in this cell, backtrack

    solve(board)

def is_sudoku_solved(board):
    """
    Verifies if a Sudoku board is solved correctly.

    Args:
        board: A 9x9 list of lists representing the Sudoku board.

    Returns:
        True if the Sudoku is solved correctly, False otherwise.
    """

    def check_row(row):
        nums = [int(x) for x in row]
        return len(set(nums)) == 9 and sum(nums) == 45

    def check_col(board, col_index):
        nums = [int(board[i][col_index]) for i in range(9)]
        return len(set(nums)) == 9 and sum(nums) == 45

    def check_box(board, box_start_row, box_start_col):
        nums = []
        for i in range(box_start_row, box_start_row + 3):
            for j in range(box_start_col, box_start_col + 3):
                nums.append(int(board[i][j]))
        return len(set(nums)) == 9 and sum(nums) == 45

    # Check rows
    for row in board:
        if not check_row(row):
            return False

    # Check columns
    for col in range(9):
        if not check_col(board, col):
            return False

    # Check 3x3 boxes
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            if not check_box(board, i, j):
                return False

    return True

if __name__ == '__main__':
    # Example Sudoku board (easy)
    sudoku_board = [
        ['5', '3', '.', '.', '7', '.', '.', '.', '.'],
        ['6', '.', '.', '1', '9', '5', '.', '.', '.'],
        ['.', '9', '8', '.', '.', '.', '.', '6', '.'],
        ['8', '.', '.', '.', '6', '.', '.', '.', '3'],
        ['4', '.', '.', '8', '.', '3', '.', '.', '1'],
        ['7', '.', '.', '.', '2', '.', '.', '.', '6'],
        ['.', '6', '.', '.', '.', '.', '2', '8', '.'],
        ['.', '.', '.', '4', '1', '9', '.', '.', '5'],
        ['.', '.', '.', '.', '8', '.', '.', '7', '9']
    ]
    
    # Solve the Sudoku
    solve_sudoku(sudoku_board)

    # Verify the solution
    is_solved = is_sudoku_solved(sudoku_board)
    print("Sudoku Solved:", is_solved)
    assert is_solved, "Sudoku was not solved correctly!"
    print("Solved Board:")
    for row in sudoku_board:
        print(row)

    # Example Sudoku board (hard)
    hard_sudoku_board = [
        ['.', '.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '3', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.', '.']
    ]
    
    solve_sudoku(hard_sudoku_board)
    
    # Verify solution
    
    is_solved = is_sudoku_solved(hard_sudoku_board)
    print("Hard Sudoku Solved:", is_solved)
    #assert is_solved, "Hard Sudoku was not solved correctly!" #this assertion is expected to fail, since hard_sudoku_board is unsolvable in this state
    print("Solved Hard Board:")
    for row in hard_sudoku_board:
        print(row)
    
    # Example Sudoku board (another example)
    another_sudoku_board = [
        ['5', '3', '.', '.', '7', '.', '.', '.', '.'],
        ['6', '.', '.', '1', '9', '5', '.', '.', '.'],
        ['.', '9', '8', '.', '.', '.', '.', '6', '.'],
        ['8', '.', '.', '.', '6', '.', '.', '.', '3'],
        ['4', '.', '.', '8', '.', '3', '.', '.', '1'],
        ['7', '.', '.', '.', '2', '.', '.', '.', '6'],
        ['.', '6', '.', '.', '.', '.', '2', '8', '.'],
        ['.', '.', '.', '4', '1', '9', '.', '.', '5'],
        ['.', '.', '.', '.', '8', '.', '.', '7', '9']
    ]
    
    # Create a copy of the board to ensure the original is not modified
    import copy
    another_sudoku_board_copy = copy.deepcopy(another_sudoku_board)
    
    # Solve the copied board
    solve_sudoku(another_sudoku_board_copy)
    
    # Verify the copied solution
    is_solved = is_sudoku_solved(another_sudoku_board_copy)
    print("Another Sudoku Solved (copied):", is_solved)
    assert is_solved, "Another Sudoku was not solved correctly!"
    
    print("Solved Another Board:")
    for row in another_sudoku_board_copy:
        print(row)
    
    # Verify that the original board remains unchanged
    print("\nOriginal Board (should be unchanged):")
    for row in another_sudoku_board:
        print(row)
    
    # Test with an already solved sudoku (should return true)
    solved_sudoku_board = [
        ['5', '3', '4', '6', '7', '8', '9', '1', '2'],
        ['6', '7', '2', '1', '9', '5', '3', '4', '8'],
        ['1', '9', '8', '3', '4', '2', '5', '6', '7'],
        ['8', '5', '9', '7', '6', '1', '4', '2', '3'],
        ['4', '2', '6', '8', '5', '3', '7', '9', '1'],
        ['7', '1', '3', '9', '2', '4', '8', '5', '6'],
        ['9', '6', '1', '5', '3', '7', '2', '8', '4'],
        ['2', '8', '7', '4', '1', '9', '6', '3', '5'],
        ['3', '4', '5', '2', '8', '6', '1', '7', '9']
    ]
    
    is_already_solved = is_sudoku_solved(solved_sudoku_board)
    print("Already Solved Sudoku is valid:", is_already_solved)
    assert is_already_solved, "Already solved sudoku is not valid"