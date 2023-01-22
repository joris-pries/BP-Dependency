# The bp_dependency package

bp_dependency is a Python package for determining the Berkelmans-Pries dependency of random variable (RV) `Y` on RV `X`.

## Paper

This package is an implementation of the ideas from 'The Dependency Function: a Generic Measure of Dependence between Random Variables', where we introduce a new dependency function. First, it is discussed what ideal properties for a dependency function are. Then, it is shown that no commonly used dependency function satisfies these requirements. We introduce a new dependency function and prove that it does satisfy all requirements, making it an ideal candidate to use as dependency function.

### Citation

If you have used the bp_dependency package, please also cite: https://arxiv.org/abs/2203.12329

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the package

```bash
pip install bp_dependency
```

---

### Windows users

```bash
python -m pip install bp_dependency
```

<!-- ```bash
python -m pip install bp_dependency
``` -->

or

```bash
py -m pip install bp_dependency
```

<!-- ```bash
py -m pip install bp_dependency
``` -->

## How to use:

The Berkelmans-Pries dependency of `Y` on `X` (notation Dep(Y|X)) is defined by `UD(X,Y) / UD(Y,Y)`, where `UD` is called the 'unordered dependency function'. For the theoretical formulation, see our paper. This package provides three functions: `unordered_bp_dependency`, `bp_dependency` and `bin_data`. We will now explain each function.

### unordered_bp_dependency

This function is used in determining the Berkelmans-Pries dependency. If `Y` does not change, it could be useful to only determine `UD(Y,Y)` once, if one wants to measure the dependency for different `X` variables.

#### Input

* `dataset` (array_like): MxK array containing M samples of K variables.
* `X_indices` (array_like): 1-dimensional list /numpy.ndarray containing the indices for the X variable.
* `Y_indices` (array_like): 1-dimensional list / numpy.ndarray containing the indices for the Y variable.
* `binning_indices` (array_like, optional): 1-dimensional list / numpy.ndarray containing the indices that need to be binned. Default is `None`, which means that no variables are binned.
* `binning_strategy` (dictionary or number or str, optional): Default is `auto`. See numpy.histogram_bin_edges. Input a dictionary if for each binning index a specific strategy should be applied.
* `midway_binning` (bool, optional): Determines if the dataset is binned using the index of the bin (False) or the midway of the bin (True). Default is False.
* `format_input` (bool, optional): Default is True. If False, no additional checks are done for the input.

#### Output

The function `unordered_bp_dependency` gives the following output:

* `float`: The unordered Berkelmans-Pries dependency score of `Y` and `X`.

#### Example

Let the dataset be given by

| X | Y |
| :-: | :-: |
| 0 | 0 |
| 1 | 1 |
| 0 | 2 |
| 1 | 3 |

where each row is as likely to be drawn. Then, `UD(X,Y)` can be determined by:

```python
 X_indices, Y_indices, dataset = (np.array([0]), np.array([1]), np.array([[0,0], [1,1], [0,2],[1,3]]))
 print(unordered_bp_dependency(dataset= dataset, X_indices= X_indices, Y_indices= Y_indices))
```

with output:

```python
 1.0
```

### bp_dependency

This function is used to determine the Berkelmans-Pries dependency (Dep(Y|X)) of `Y` on `X`. If `Y` is constant, a default value of -1.0 is returned.

#### Input

* `dataset` (array_like): MxK array containing M samples of K variables.
* `X_indices` (array_like): 1-dimensional list /numpy.ndarray containing the indices for the X variable.
* `Y_indices` (array_like): 1-dimensional list / numpy.ndarray containing the indices for the Y variable.
* `binning_indices` (array_like, optional): 1-dimensional list / numpy.ndarray containing the indices that need to be binned. Default is `None`, which means that no variables are binned.
* `binning_strategy` (dictionary or number or str, optional): Default is `auto`. See numpy.histogram_bin_edges. Input a dictionary if for each binning index a specific strategy should be applied.
* `midway_binning` (bool, optional): Determines if the dataset is binned using the index of the bin (False) or the midway of the bin (True). Default is False.
* `format_input` (bool, optional): Default is True. If False, no additional checks are done for the input.

#### Output

The function `bp_dependency` gives the following output:

* `float`: The Berkelmans-Pries dependency score of `Y` on `X`. If `Y` is constant, -1.0 is returned.

#### Example

Let the dataset be given by

| X | Y |
| :-: | :-: |
| 0 | 0 |
| 1 | 1 |
| 0 | 2 |
| 1 | 3 |

where each row is as likely to be drawn. Then, `Dep(Y|X)` can be determined by:

```python
 X_indices, Y_indices, dataset = (np.array([0]), np.array([1]), np.array([[0,0], [1,1], [0,2],[1,3]]))
 print(bp_dependency(dataset= dataset, X_indices= X_indices, Y_indices= Y_indices))
```

with output:

```python
 0.6666666666666666
```

### bin_data

This function is used to bin variables. It is used in `unordered_dependency`. In order to avoid repetitive binning, one should consider applying this function first.

#### Input

* `x` (array_like): 1-dimensional list/numpy.ndarray containing the values that need to be binned.
* `bins` (int or sequence of scalars or str, optional): Default is `auto`. See numpy.histogram_bin_edges for more information.
* `Y_indices` (array_like): 1-dimensional list / numpy.ndarray containing the indices for the Y variable.
* `rrange` ((float, float), optional): Default is `None`. It is the lower and upper range of the bins. `None` simply determines the minimum and maximum of `x` as range.
* `midways` (bool, optional): Determines if the values are reduced to the midways of the bins (if True) or just the index of the bins (if False).

#### Output

The function `bin_data` gives the following output:

* `numpy.ndarray`: The binned data

#### Example

Let the dataset be given by

| X |
| :-: |
| 0 |
| 1 |
| 2 |
| 3 |
| 5 |
| 6 |
| 7 |
| 8 |

where each row is as likely to be drawn. This dataset can then be binned by:

```python
x = [0, 1, 2, 3, 5, 6, 7, 8]
print(bin_data(x= x, bins= 2, midways=True))
```

 with output

```python
 [2. 2. 2. 2. 6. 6. 6. 6.]
```

<!--- een concreet voorbeeld geven waarbij alledrie de functies gebruikt worden --->

## License

[MIT](https://choosealicense.com/licenses/mit/)
