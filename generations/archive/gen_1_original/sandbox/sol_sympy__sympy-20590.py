from sympy import *
from sympy.matrices import MatrixSymbol

def test_matrix_simplification():
    """
    Tests the simplification of matrix expressions, focusing on Transpose and Inverse operations.
    """

    n = symbols('n', integer=True, positive=True)
    A = MatrixSymbol('A', n, n)

    # Test case 1: Transpose of a transpose should simplify to the original matrix.
    expr1 = A.T.T
    simplified_expr1 = simplify(expr1)
    expected_expr1 = A
    assert simplified_expr1 == expected_expr1, f"Test Case 1 Failed: Expected {expected_expr1}, but got {simplified_expr1}"
    print("Test Case 1 (Transpose) Passed")

    # Test case 2: Basic inverse simplification.  Inverse of an inverse.
    expr2 = A.I.I
    simplified_expr2 = simplify(expr2)
    expected_expr2 = A
    assert simplified_expr2 == expected_expr2, f"Test Case 2 Failed: Expected {expected_expr2}, but got {simplified_expr2}"
    print("Test Case 2 (Inverse) Passed")

    # Test case 3: Transpose of an inverse
    expr3 = A.I.T
    simplified_expr3 = simplify(expr3)
    expected_expr3 = A.T.I
    assert simplified_expr3 == A.T.I , f"Test Case 3 Failed: Expected {A.T.I}, but got {simplified_expr3}" #Verify equivalence
    print("Test Case 3 (Transpose of Inverse) Passed")
    
    #Test case 4: Simplify Inverse of a Matrix Symbol raised to power
    p = Symbol('p', integer = True)
    expr4 = (A**p).I
    simplified_expr4 = simplify(expr4)
    expected_expr4 = A**(-p)
    assert simplified_expr4 == expected_expr4, f"Test Case 4 Failed: Expected {expected_expr4}, but got {simplified_expr4}"
    print("Test Case 4 (Inverse of Matrix Symbol raised to power) Passed")

    print("All test cases passed!")


if __name__ == '__main__':
    test_matrix_simplification()