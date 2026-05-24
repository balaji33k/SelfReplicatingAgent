import numpy as np
from skimage.morphology import skeletonize
from skimage.util import invert
from scipy import ndimage

def verify_connectivity(skeleton):
    """
    Verifies connectivity of a skeletonized image.

    Args:
        skeleton (np.ndarray): The skeletonized binary image.

    Returns:
        bool: True if the skeleton is connected, False otherwise.
    """
    labeled_skeleton, num_labels = ndimage.label(skeleton)
    return num_labels == 1


def test_skeletonize_connectivity():
    """
    Tests the skeletonize function for connectivity on a 'thin bridge' pattern.
    """
    # Create a 'thin bridge' binary image
    image = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ], dtype=bool)

    # Skeletonize the image
    skeleton = skeletonize(image)

    # Verify connectivity
    is_connected = verify_connectivity(skeleton)

    # Assert that the skeleton is connected
    assert is_connected, "Skeleton should be connected for this pattern."

    # Create a 'thin bridge' binary image with some isolated pixels
    image_isolated = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ], dtype=bool)

    # Skeletonize the image
    skeleton_isolated = skeletonize(image_isolated)

    # Verify connectivity
    is_connected_isolated = verify_connectivity(skeleton_isolated)

    # Assert that the skeleton is connected
    assert is_connected_isolated, "Skeleton should be connected for this pattern."


    print("Connectivity test passed.")

if __name__ == "__main__":
    test_skeletonize_connectivity()