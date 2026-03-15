import pytest
import numpy as np
import cv2
import os

@pytest.fixture
def dummy_image():
    """Returns a simple 4x4 grayscale image."""
    img = np.array([
        [0, 100, 200, 255],
        [50, 150, 250, 100],
        [200, 50, 0, 150],
        [255, 200, 100, 50]
    ], dtype=np.uint8)
    return img

@pytest.fixture
def resource_path():
    """Returns the path to the resources directory."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources")
