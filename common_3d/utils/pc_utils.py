"""
Utilities for point clouds.

Code modified from Jiayuan Gu.
"""
import numpy as np


def pad_or_clip(array: np.array, n: int, fill_value=0):
    """Pad or clip an array with constant values.
    It is usually used for sampling a fixed number of points.
    """
    if array.shape[0] >= n:
        return array[:n]
    else:
        pad = np.full((n - array.shape[0],) + array.shape[1:], fill_value, dtype=array.dtype)
        return np.concatenate([array, pad], axis=0)


def pad_or_clip_v2(array: np.array, n: int):
    """Pad or clip an array with the first item.
    It is usually used for sampling a fixed number of points (PointNet and variants).
    """
    if array.shape[0] >= n:
        return array[:n]
    else:
        pad = np.repeat(array[0:1], n - array.shape[0], axis=0)
        return np.concatenate([array, pad], axis=0)


def pad(array: np.array, n: int, fill_value=None):
    """Pad an array with a constant or the first item."""
    assert array.shape[0] <= n
    if fill_value is None:
        pad = np.repeat(array[0:1], n - array.shape[0], axis=0)
    else:
        pad = np.full((n - array.shape[0],) + array.shape[1:], fill_value, dtype=array.dtype)
    return np.concatenate([array, pad], axis=0)


# ---------------------------------------------------------------------------- #
# Unit test
# ---------------------------------------------------------------------------- #
def test_pad_or_clip():
    test_cases = [(np.array([1, 2, 1]), 5), (np.array([0.5, -0.1, 1.0]), 2), (np.array([0, 1, 1], bool), 4)]
    expected_list = [np.array([1, 2, 1, 0, 0]), np.array([0.5, -0.1]), np.array([0, 1, 1, 0], bool)]
    test_cases += [(np.empty([0, 3]), 5)]
    expected_list += [np.zeros([5, 3])]
    for test_case, expected in zip(test_cases, expected_list):
        actual = pad_or_clip(*test_case)
        np.testing.assert_almost_equal(actual, expected)
