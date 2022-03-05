# bp_dependency

bp_dependency is a Python package for determining the Berkelmans-Pries dependency between `Y` and `X`.

## Paper

This package is an implementation of the ideas from 'The Dependency Function: a Generic Measure of Dependence between Random Variables', where we introduce a new dependency function. First, it is discussed what ideal properties for a dependency function are. Then, it is shown that no commonly used dependency function satisfies these requirements. We introduce a new dependency function and prove that it does satisfy all requirements, making it an ideal candidate to use as dependency function.

### Citation
If you have used the bp_dependency package, please also cite: (Our work is under review)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the package

```bash
pip install bp_dependency
```

----

### Windows users

```bash
python -m pip install --upgrade bp_dependency
```

<!-- ```bash
python -m pip install bp_dependency
``` -->

or

```bash
py -m pip install --upgrade bp_dependency
```

<!-- ```bash
py -m pip install bp_dependency
``` -->

## How to use:

The Berkelmans-Pries dependency of `Y` on `X` (notation Dep(Y|X)) is defined by `UD(X,Y) / UD(Y,Y)`, where `UD` is called the 'unordered dependency function'. For the theoretical formulation, see our paper. This package provides three functions: `bin_data`, `unordered_bp_dependency` and `bp_dependency`. We will now explain each function.



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

* `float`: The unordered Berkelmans-Pries dependency score of Y and X.


#### Example
Let the dataset be given by 
| X  | Y  |
| :-: | :-: |
| 0 | 0 |
| 1 | 1 |
| 2 | 0 |
| 3 | 1 |
where each row is as likely to be drawn. Then, `UD(X,Y)` can be determined by:

```python
 X_indices, Y_indices, dataset = (np.array([1]), np.array([0]), np.array([[0,0], [1,1], [2,0],[3,1]]))
  print(unordered_bp_dependency(dataset= dataset, X_indices= X_indices, Y_indices= Y_indices))
```
with output:
```python
 1.0
```



## License

[MIT](https://choosealicense.com/licenses/mit/)
