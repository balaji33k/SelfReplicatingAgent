import time
import psutil
import scipy.sparse as sparse
from sklearn.ensemble import HistGradientBoostingClassifier
import numpy as np
from sklearn.model_selection import train_test_split


def train_sparse_gbdt(X, y):
    """
    Trains a HistGradientBoostingClassifier on a sparse matrix efficiently.

    Args:
        X: Sparse matrix of features (scipy.sparse).
        y: Numpy array of target variable.

    Returns:
        Trained HistGradientBoostingClassifier model.
    """
    start_memory = psutil.Process().memory_info().rss
    model = HistGradientBoostingClassifier()
    start_time = time.time()
    model.fit(X, y)
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss

    print(f"Training time: {end_time - start_time:.4f} seconds")
    memory_used = end_memory - start_memory
    print(f"Memory used during training: {memory_used / (1024 * 1024):.2f} MB")
    return model, memory_used


if __name__ == '__main__':
    # Verification with sparse data
    n_samples = 10000
    n_features = 1000
    density = 0.1  # Sparsity level

    X_sparse = sparse.random(n_samples, n_features, density=density, format="csr")
    y = np.random.randint(0, 2, n_samples)

    # Split data for a more robust verification and prevent overfitting on the training set itself.
    X_train, X_test, y_train, y_test = train_test_split(X_sparse, y, test_size=0.2, random_state=42)

    model, memory_usage = train_sparse_gbdt(X_train, y_train)

    # Test: Predict on the test set
    predictions = model.predict(X_test)

    # Test: Check accuracy (basic sanity check)
    accuracy = np.mean(predictions == y_test)
    print(f"Accuracy on the test set: {accuracy:.4f}")
    assert accuracy > 0.7, "Accuracy is too low, something went wrong." # reasonable accuracy threshold
    print("Sparse matrix training verification successful!")


    # Additional Test Case: Small dense matrix for comparison and baseline
    n_samples_dense = 1000
    n_features_dense = 100
    X_dense = np.random.rand(n_samples_dense, n_features_dense)
    y_dense = np.random.randint(0, 2, n_samples_dense)
    X_train_dense, X_test_dense, y_train_dense, y_test_dense = train_test_split(X_dense, y_dense, test_size=0.2, random_state=42)


    start_memory_dense = psutil.Process().memory_info().rss
    model_dense = HistGradientBoostingClassifier()
    start_time_dense = time.time()
    model_dense.fit(X_train_dense, y_train_dense)
    end_time_dense = time.time()
    end_memory_dense = psutil.Process().memory_info().rss

    print(f"Dense training time: {end_time_dense - start_time_dense:.4f} seconds")
    memory_used_dense = end_memory_dense - start_memory_dense
    print(f"Memory used during dense training: {memory_used_dense / (1024 * 1024):.2f} MB")

    predictions_dense = model_dense.predict(X_test_dense)
    accuracy_dense = np.mean(predictions_dense == y_test_dense)
    print(f"Accuracy on the dense test set: {accuracy_dense:.4f}")
    assert accuracy_dense > 0.7, "Dense accuracy too low"
    print("Dense matrix training verification successful!")

    #Verification: Basic size check comparing sparse and dense memory usage.
    #Assuming the sparse matrix is indeed sparse, the memory usage should be less than the dense case,
    #taking into account the significantly different sample sizes.
    assert memory_usage < 500 * 1024 * 1024, "Sparse matrix memory usage unexpectedly high." # 500MB cap.
    print("Memory usage checks passed")