import pandas as pd
import numpy as np

def test_multiindex_duplicate_slicing():
    """
    Tests slicing a DataFrame with a MultiIndex containing duplicate labels on one level.
    """

    # Create a MultiIndex with duplicates in the first level
    index = pd.MultiIndex.from_tuples(
        [('A', 1), ('A', 2), ('B', 3), ('B', 4), ('A', 5), ('A', 6)],
        names=['letter', 'number']
    )

    # Create a DataFrame with the MultiIndex
    df = pd.DataFrame(np.random.randn(6, 3), index=index, columns=['col1', 'col2', 'col3'])

    # Slice the DataFrame based on the duplicated level ('letter')
    sliced_df = df.loc['A']

    # Verify the shape of the sliced DataFrame
    expected_shape = (4, 3)  # Expecting 4 rows (A appears 4 times) and 3 columns
    actual_shape = sliced_df.shape

    print(f"Original DataFrame:\n{df}")
    print(f"\nSliced DataFrame (df.loc['A']):\n{sliced_df}")
    print(f"\nExpected Shape: {expected_shape}")
    print(f"Actual Shape: {actual_shape}")


    assert actual_shape == expected_shape, f"Shape mismatch. Expected {expected_shape}, got {actual_shape}"

    # Additional test case: check index values
    expected_index_values = [1, 2, 5, 6]
    actual_index_values = sliced_df.index.tolist()

    assert actual_index_values == expected_index_values, f"Index values mismatch. Expected {expected_index_values}, got {actual_index_values}"


    print("MultiIndex duplicate slicing test passed.")


if __name__ == "__main__":
    test_multiindex_duplicate_slicing()