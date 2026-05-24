```python
import numpy as np
from sklearn.impute import SimpleImputer
import pytest
import warnings

def test_simple_imputer_axis_deprecation():
    """
    Test that passing axis=1 to SimpleImputer raises a DeprecationWarning.
    """
    with pytest.warns(DeprecationWarning, match="The 'axis' parameter of SimpleImputer"):
        SimpleImputer(axis=1)


def test_simple_imputer_default_behavior():
    """
    Test that SimpleImputer works correctly with default settings (axis=0)
    on a 2D array with NaNs.
    """
    X = np.array([[np.nan, 1, 2], [3, np.nan, 4], [5, 6, np.nan]])
    imputer = SimpleImputer()  # Default strategy is 'mean' and axis=0 is implicit
    X_imputed = imputer.fit_transform(X)

    expected_imputed = np.array(
        [[4., 1., 2.],
         [3., 3.5, 4.],
         [5., 6., 3.]])

    np.testing.assert_allclose(X_imputed, expected_imputed)
    assert imputer.statistics_.shape == (3,)
    np.testing.assert_allclose(imputer.statistics_, np.array([4., 3.5, 3.]))


def test_simple_imputer_axis_0_explicit():
    """
    Test that SimpleImputer works correctly with axis=0 specified
    on a 2D array with NaNs.
    """
    X = np.array([[np.nan, 1, 2], [3, np.nan, 4], [5, 6, np.nan]])
    imputer = SimpleImputer(axis=0)
    X_imputed = imputer.fit_transform(X)

    expected_imputed = np.array(
        [[4., 1., 2.],
         [3., 3.5, 4.],
         [5., 6., 3.]])

    np.testing.assert_allclose(X_imputed, expected_imputed)
    assert imputer.statistics_.shape == (3,)
    np.testing.assert_allclose(imputer.statistics_, np.array([4., 3.5, 3.]))


def test_simple_imputer_with_axis_1():
     """
     Test if SimpleImputer with axis=1 specified raises a deprecation warning and
     the data is correctly imputed as before (along the columns, but still returning
     a consistent shape as if operating row-wise). It implicitly converts column imputations to
     row-wise imputations under the hood.
     """
     X = np.array([[np.nan, 1, 2], [3, np.nan, 4], [5, 6, np.nan]])
     with pytest.warns(DeprecationWarning, match="The 'axis' parameter of SimpleImputer"):
         imputer = SimpleImputer(axis=1)
         X_imputed = imputer.fit_transform(X)

     expected_imputed = np.array(
         [[1.5, 1., 2.],
          [3., 3.5, 4.],
          [5., 6., 5.5]])


     np.testing.assert_allclose(X_imputed, expected_imputed)
     assert imputer.statistics_.shape == (3,)
     np.testing.assert_allclose(imputer.statistics_, np.array([1.5, 3.5, 5.5]))


if __name__ == "__main__":
    pytest.main([__file__])
```