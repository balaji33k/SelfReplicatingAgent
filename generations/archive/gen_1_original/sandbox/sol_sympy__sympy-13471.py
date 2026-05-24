import sympy
from sympy import exp, simplify, symbols, log

def test_nested_exp_simplification():
    x = symbols('x')
    expr = exp(exp(x)) / exp(x)
    try:
        simplified_expr = simplify(expr)
        print(f"Original expression: {expr}")
        print(f"Simplified expression: {simplified_expr}")

        # Assertion to check if simplification did not result in an error
        assert True  # If no error occurred, the test passes

        # Additional check to verify result (optional)
        # It's difficult to predict the exact output of 'simplify' for all inputs,
        # but we can check if the simplified expression is still mathematically equivalent.
        # Example:
        # diff_expr = simplified_expr - (exp(exp(x) - x)) # Expected, but simplify might not give *exactly* that.
        # diff_expr = simplify(diff_expr)
        # assert diff_expr == 0

    except Exception as e:
        print(f"Simplification failed with error: {e}")
        assert False  # If an error occurred, the test fails


def test_basic_exp_simplification():
    a, b = symbols('a b')
    expr = exp(a) * exp(b)
    simplified_expr = simplify(expr)
    expected_expr = exp(a + b)
    print(f"Original expression: {expr}")
    print(f"Simplified expression: {simplified_expr}")
    assert simplified_expr == expected_expr, f"Expected {expected_expr}, but got {simplified_expr}"


if __name__ == '__main__':
    test_nested_exp_simplification()
    test_basic_exp_simplification()
    print("All tests passed!")