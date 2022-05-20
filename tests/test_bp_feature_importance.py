import pytest
import numpy as np
import sys
sys.path.append('E:/OneDrive/PhD/GitHub/Official_Feature_Importance/src/bp_feature_importance')

from all_fi_measures import *


# def test_length_list_of_all_methods():
#     assert len(list_of_all_methods) == 247


 
# @pytest.mark.parametrize("name", list_of_all_methods)
# def test_determine_fi(name):
#     kwargs, fi_method_name = initialize_experiment_variables(name, data_path = 'E:/OneDrive/PhD/GitHub/Official_Feature_Importance/src/bp_feature_importance/datasets/decimal_system.pickle')
#     assert len(determine_fi(fi_method_name, data_path = 'E:/OneDrive/PhD/GitHub/Official_Feature_Importance/src/bp_feature_importance/datasets/decimal_system.pickle', **kwargs)) == 3

# @pytest.mark.parametrize("Y_indices, dataset, expected", [
#     ([0], np.array([[1,2], [3,4], [5,6]]), False),
#     ([0], np.array([[1,2], [1,4], [1,6]]), True),
# ])
# def test_is_Y_constant(Y_indices, dataset, expected):
#     assert is_Y_constant(Y_indices, dataset) == expected


# @pytest.mark.parametrize("x, bins, expected", [
#     (np.array([1,2,3,4]), 4, np.array([0, 1, 2, 3])),
#     (4 * np.array([1,2,3,4]), 4, np.array([0, 1, 2, 3])),
#     (np.array([1,2,3,4]), 2, np.array([0, 0, 1, 1])),
#     (np.array([1,2,3,4]), 7, np.array([0, 2, 4, 6])),
# ])
# def test_bin(x, bins, expected):
#     assert np.array_equal(bin_data(x=x, bins=bins, midways= False), expected)

# @pytest.mark.parametrize("Y_indices, X_indices, dataset, binning_indices, binning_strategy", [
#     ([0], [1,2], np.array([[1,0,0], [1,0,1],[1,1,0],[1,1,1]]), None, None),
#     ([0], [1,2], np.random.uniform(size = (5,6)), [0], 1),
# ])
# def test_constant_Y(Y_indices, X_indices, dataset, binning_indices, binning_strategy):
#     assert bp_dependency(Y_indices= Y_indices, X_indices= X_indices, dataset= dataset, binning_indices= binning_indices, binning_strategy= binning_strategy) == -1.0


# @pytest.mark.parametrize("X_indices, Y_indices, dataset, expected", [
#     (np.array([1]), np.array([0]),np.array([[0,0], [1,1], [0,0],[1,1]]), 1.0),
#     (np.array([1]), np.array([0]),np.array([[0,0], [1,1], [2,2],[3,3]]), 1.5),
#     (np.array([1]), np.array([0]),np.array([[0,0], [1,1], [2,0],[3,1]]), 1.0),
# ])
# def test_unordered_bp_dependency(X_indices, Y_indices, dataset,expected):
#     assert unordered_bp_dependency(X_indices= X_indices, Y_indices= Y_indices, dataset= dataset) == expected

