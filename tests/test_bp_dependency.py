import pytest
import numpy as np
from bp_dependency.dependency import is_Y_constant, bin_data, unordered_bp_dependency, bp_dependency




@pytest.mark.parametrize("Y_indices, dataset, expected", [
    ([0], np.array([[1,2], [3,4], [5,6]]), False),
    ([0], np.array([[1,2], [1,4], [1,6]]), True),
])
def test_is_Y_constant(Y_indices, dataset, expected):
    assert is_Y_constant(Y_indices, dataset) == expected


@pytest.mark.parametrize("x, bins, expected", [
    (np.array([1,2,3,4]), 4, np.array([0, 1, 2, 3])),
    (4 * np.array([1,2,3,4]), 4, np.array([0, 1, 2, 3])),
    (np.array([1,2,3,4]), 2, np.array([0, 0, 1, 1])),
    (np.array([1,2,3,4]), 7, np.array([0, 2, 4, 6])),
])
def test_bin(x, bins, expected):
    assert np.array_equal(bin_data(x=x, bins=bins, midways= False), expected)

@pytest.mark.parametrize("Y_indices, X_indices, dataset, binning_indices, binning_strategy", [
    ([0], [1,2], np.array([[1,0,0], [1,0,1],[1,1,0],[1,1,1]]), None, None),
    ([0], [1,2], np.random.uniform(size = (5,6)), [0], 1),
])
def test_constant_Y(Y_indices, X_indices, dataset, binning_indices, binning_strategy):
    assert bp_dependency(Y_indices= Y_indices, X_indices= X_indices, dataset= dataset, binning_indices= binning_indices, binning_strategy= binning_strategy) == -1.0


@pytest.mark.parametrize("X_indices, Y_indices, dataset, expected", [
    (np.array([1]), np.array([0]),np.array([[0,0], [1,1], [0,0],[1,1]]), 1.0),
    (np.array([1]), np.array([0]),np.array([[0,0], [1,1], [2,2],[3,3]]), 1.5),
    (np.array([1]), np.array([0]),np.array([[0,0], [1,1], [2,0],[3,1]]), 1.0),
])
def test_unordered_bp_dependency(X_indices, Y_indices, dataset,expected):
    assert unordered_bp_dependency(X_indices= X_indices, Y_indices= Y_indices, dataset= dataset) == expected

