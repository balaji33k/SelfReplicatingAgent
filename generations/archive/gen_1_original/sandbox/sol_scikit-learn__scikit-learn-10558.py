import numpy as np
from sklearn.impute import SimpleImputer
import warnings
import unittest

class TestSimpleImputerDeprecation(unittest.TestCase):
    def test_axis_deprecation(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # Ensure warnings are not ignored
            imp = SimpleImputer(strategy="mean", axis=1)
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, DeprecationWarning))
            self.assertIn("'axis' is deprecated in version 1.2 and will be removed in", str(w[-1].message))

    def test_imputation_with_default_axis(self):
        X = np.array([[np.nan, 2, np.nan, 0],
                      [3, 4, np.nan, 1],
                      [np.nan, np.nan, np.nan, 5],
                      [np.nan, 3, np.nan, 4]], dtype=np.float64)
        imp = SimpleImputer(strategy="mean")
        X_imputed = imp.fit_transform(X)

        X_expected = np.array([[3., 2., np.nan, 0.],
                              [3., 4., np.nan, 1.],
                              [3., 3., np.nan, 5.],
                              [3., 3., np.nan, 4.]])

        mean_0 = np.nanmean(X[:, 0])
        mean_1 = np.nanmean(X[:, 1])
        mean_2 = np.nanmean(X[:, 2])
        mean_3 = np.nanmean(X[:, 3])

        X_expected = np.array([[mean_0, 2, mean_2, 0],
                              [3, 4, mean_2, 1],
                              [mean_0, mean_1, mean_2, 5],
                              [mean_0, 3, mean_2, 4]])
        
        mean_vector = np.nanmean(X, axis=0)
        
        X_filled = np.where(np.isnan(X), mean_vector, X)
        
        self.assertTrue(np.allclose(X_imputed, X_filled))

    def test_imputation_with_axis_1_deprecated(self):
        X = np.array([[np.nan, 2, np.nan, 0],
                      [3, 4, np.nan, 1],
                      [np.nan, np.nan, np.nan, 5],
                      [np.nan, 3, np.nan, 4]], dtype=np.float64)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            imp = SimpleImputer(strategy="mean", axis=1)
            X_imputed = imp.fit_transform(X)
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, DeprecationWarning))
        
        mean_vector = np.nanmean(X, axis=0)
        
        X_filled = np.where(np.isnan(X), mean_vector, X)
        
        self.assertTrue(np.allclose(X_imputed, X_filled))


if __name__ == '__main__':
    unittest.main()