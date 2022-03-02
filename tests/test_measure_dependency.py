import pytest
import numpy as np
from measure_dependency.dependency import dependency, bin_data, unordered_dependency



@pytest.mark.parametrize("Y_indices, X_indices", [
    ([0,1], [2,7]),
    ([0,1], [2,-1]),
    ([2,7], [0,1]),
    ([2,-1], [0,1]),
])
def test_invalid_indices(Y_indices, X_indices):
    with pytest.raises(ValueError):
        dependency(Y_indices=Y_indices, X_indices = X_indices, dataset= np.random.uniform(size = (5,6)))

@pytest.mark.parametrize("x, bins, expected", [
    (np.array([1,2,3,4]), 4, np.array([0, 1, 2, 3])),
    (4 * np.array([1,2,3,4]), 4, np.array([0, 1, 2, 3])),
    (np.array([1,2,3,4]), 2, np.array([0, 0, 1, 1])),
    (np.array([1,2,3,4]), 7, np.array([0, 2, 4, 6])),
])
def test_bin(x, bins, expected):
    assert np.array_equal(bin_data(x, bins), expected)


@pytest.mark.parametrize("Y_indices, X_indices, dataset, binning_indices, binning_strategy", [
    ([0], [1,2], np.array([[1,0,0], [1,0,1],[1,1,0],[1,1,1]]), None, None),
    ([0], [1,2], np.random.uniform(size = (5,6)), [0], 1),
])
def test_constant(Y_indices, X_indices, dataset, binning_indices, binning_strategy):
    assert dependency(Y_indices= Y_indices, X_indices= X_indices, dataset= dataset, binning_indices= binning_indices, binning_strategy= binning_strategy) == 0.0


@pytest.mark.parametrize("X_discrete_indices, Y_discrete_indices, local_dataset, expected", [
    (np.array([1]), np.array([0]),np.array([[0,0], [1,1], [0,0],[1,1]]), 1.0),
    (np.array([1]), np.array([0]),np.array([[0,0], [1,1], [2,2],[3,3]]), 1.5),
    (np.array([1]), np.array([0]),np.array([[0,0], [1,1], [2,0],[3,1]]), 1.0),
])
def test_ud_only_discrete(X_discrete_indices, Y_discrete_indices, local_dataset,expected):
    assert unordered_dependency(X_discrete_indices= X_discrete_indices, X_continuous_indices= None, Y_discrete_indices=Y_discrete_indices, Y_continuous_indices= None, local_dataset=local_dataset, kde_strategy= None) == expected


# @pytest.mark.parametrize("test_input,expected", [
#     ('ll', 2.5),
#     ('dd', 2.5),
# ])
# def test_multi_slaps(test_input, expected):
#     assert dependency(test_input) == expected


# def test_invalid_indices():
#     with pytest.raises(ValueError):
#         dependency(Y_indices=[0,1,23], X_indices=[4,5], dataset= np.random.uniform(size = (5,6)))

# def test_empty_slap():
#     assert slap_many(LikeState.empty, '') is LikeState.empty


# def test_single_slaps():
#     assert slap_many(LikeState.empty, 'l') is LikeState.liked
#     assert slap_many(LikeState.empty, 'd') is LikeState.disliked


# @pytest.mark.parametrize("test_input,expected", [
#     ('ll', LikeState.empty),
#     ('dd', LikeState.empty),
#     ('ld', LikeState.disliked),
#     ('dl', LikeState.liked),
#     ('ldd', LikeState.empty),
#     ('lldd', LikeState.empty),
#     ('ddl', LikeState.liked),
# ])
# def test_multi_slaps(test_input, expected):
#     assert slap_many(LikeState.empty, test_input) is expected


# @pytest.mark.skip(reason="regexes not supported yet")
# def test_regex_slaps():
#     assert slap_many(LikeState.empty, '[ld]*ddl') is LikeState.liked


# @pytest.mark.xfail
# def test_divide_by_zero():
#     assert 1 / 0 == 1


# def test_invalid_slap():
#     with pytest.raises(ValueError):
#         slap_many(LikeState.empty, 'x')


# @pytest.mark.xfail
# def test_db_slap(db_conn):
#     db_conn.read_slaps()
#     assert ...


# def test_print(capture_stdout):
#     print("hello")
#     assert capture_stdout["stdout"] == "hello\n"

# # def test_many_slaps():
# #     assert slap_many(LikeState.empty, 'll') is LikeState.empty
# #     assert slap_many(LikeState.empty, 'dd') is LikeState.empty
# #     assert slap_many(LikeState.empty, 'ld') is LikeState.disliked
# #     assert slap_many(LikeState.empty, 'dl') is LikeState.liked
# #     assert slap_many(LikeState.empty, 'ldd') is LikeState.empty
# #     assert slap_many(LikeState.empty, 'lldd') is LikeState.empty
# #     assert slap_many(LikeState.empty, 'ddl') is LikeState.liked
