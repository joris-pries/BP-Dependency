# %%
import numpy as np
import pandas as pd

__all__ = ['bin_data', 'unordered_bp_dependency', 'bp_dependency']

def is_Y_constant(Y_indices, dataset) -> bool:
    """
    This function is used to check if Y = dataset[:, Y_indices] is constant, as this leads to a special case for dependency functions.
    """
    Y= dataset[:, Y_indices]
    unique_rows_Y = np.unique(Y, axis = 0)
    if unique_rows_Y.shape[0] == 1:
        return True
    else:
        return False

# Function used to bin the data
def bin_data(x, bins= 'auto', rrange= None, midways= True):
    """
    This function is used to bin data.

    Args:
    --------
        x (array_like): list/numpy.ndarray containing the values that need to be binned.
        bins (int or sequence of scalars or str, optional): Default is `auto`. See numpy.histogram_bin_edges for more information.
        rrange ((float, float), optional): Default is `None`. It is the lower and upper range of the bins. `None` simply determines the minimum and maximum of `x` as range.
        midways (bool, optional): Determines if the values are reduced to the midways of the bins (if True) or just the index of the bins (if False).

    Returns:
    --------
        numpy.ndarray: The binned data.

    See also:
    --------
        numpy.histogram_bin_edges, bp_dependency, unordered_bp_dependency

    Example:
    --------
        >>> x = [0, 1, 2, 3, 5, 6, 7, 8]
        >>> print(bin_data(x= x, bins= 2, midways=True))
        [2. 2. 2. 2. 6. 6. 6. 6.]
        >>> print(bin_data(x= x, bins= 2, midways=False))
        [0 0 0 0 1 1 1 1]
        >>> print(bin_data(x= x, bins= 2, rrange = (0,12), midways=False))
        [0 0 0 0 0 0 1 1]
    """

    if not isinstance(x, np.ndarray):
        x = np.array(x)

    if rrange is None:
        rrange = (np.amin(x), np.amax(x))
    x_bin_edges = np.histogram_bin_edges(x, bins= bins, range= rrange)
    x_digitized = np.digitize(x, bins = x_bin_edges, right = True) - 1
    x_digitized[x_digitized == -1] = 0

    if midways:
        # We do need the original data. The values are converted to the midway of the corresponding bin
        midways_bins = [(x_bin_edges[i] + x_bin_edges[i + 1]) / 2 for i in range(len(x_bin_edges) - 1)]
        x_converted = x.copy().astype(np.float)
        for i, digit in enumerate(x_digitized):
            x_converted[i] = midways_bins[digit]

    if not midways:
        # We do not need the original data. We only need the bin number
        x_converted = x_digitized

    return x_converted


# %%

# Function to get the probabibility density function
def convert_variable_to_prob_density_function(x):
    """
    This function converts `x` into a dictionary and counts how often each row occurs.
    """

    return(pd.DataFrame(x).value_counts(normalize=True).to_dict())

# %%
def format_input_variables(**kwargs):
    """
    This function is used to convert some input variables into the right format to make the `bp_dependency` and `bp_unordered_dependency` more convenient to use.
    """

    if 'dataset' in kwargs:
        kwargs['dataset'] = np.asarray(kwargs['dataset'])

    if 'X_indices' in kwargs:
        kwargs['X_indices'] = np.asarray(kwargs['X_indices'])

    if 'Y_indices' in kwargs:
        kwargs['Y_indices'] = np.asarray(kwargs['Y_indices'])

    if 'binning_indices' in kwargs:
        if kwargs['binning_indices'] is not None:
            kwargs['binning_indices'] = np.asarray(kwargs['binning_indices'])

    if 'binning_strategy' in kwargs and kwargs['binning_indices'] is not None:
        if not isinstance(kwargs['binning_strategy'], dict):
            strat = kwargs['binning_strategy']
            kwargs['binning_strategy'] = {bin_index : strat for bin_index in kwargs['binning_indices']}

    return(kwargs.values())



# %%
def unordered_bp_dependency(dataset, X_indices, Y_indices, binning_indices= None, binning_strategy = 'auto', midway_binning = False, format_input = True) -> float:

    """
    This function is used for the `dependency' function and determines the unordered Berkelmans-Pries dependency of Y and X (notation: UD(Y,X)) for a given dataset.

    Args:
    --------
        dataset (array_like): MxK array containing M samples of K variables.
        X_indices (array_like): 1-dimensional list /numpy.ndarray containing the indices for the X variable.
        Y_indices (array_like): 1-dimensional list / numpy.ndarray containing the indices for the Y variable.
        binning_indices (array_like, optional): 1-dimensional list / numpy.ndarray containing the indices that need to be binned. Default is `None`, which means that no variables are binned.
        binning_strategy (dictionary or number or str, optional): Default is `auto`. See numpy.histogram_bin_edges. Input a dictionary if for each binning index a specific strategy should be applied.
        midway_binning (bool, optional): Determines if the dataset is binned using the index of the bin (False) or the midway of the bin (True). Default is False.
        format_input (bool, optional): Default is True. If False, no additional checks are done for the input.


    Returns:
    --------
        float: The unordered Berkelmans-Pries dependency score of Y and X.

    Raises:
    --------


    See also:
    --------
        bp_dependency, bin_data, numpy.histogram_bin_edges

    Example:
    --------
        >>> X_indices, Y_indices, dataset = (np.array([0]), np.array([1]), np.array([[0,0], [1,1], [0,2],[1,3]]))
        >>> print(unordered_bp_dependency(dataset= dataset, X_indices= X_indices, Y_indices= Y_indices))
        1.0
    """


    if format_input:
        # format variables
        dataset, X_indices, Y_indices, binning_indices, binning_strategy = format_input_variables(dataset= dataset, X_indices= X_indices, Y_indices= Y_indices, binning_indices= binning_indices, binning_strategy= binning_strategy)

    # copy dataset to local dataset
    local_dataset = dataset.copy()

    # Binning all binning_indices
    if binning_indices is not None:
        for bin_index in binning_indices:
            local_dataset[:, bin_index] = bin_data(x= local_dataset[:, bin_index], bins = binning_strategy[bin_index], midways = midway_binning)


    # initialization dep variable
    dep = 0

    # determining prob. density of X
    X_d = local_dataset[:, X_indices]
    p_X_d = convert_variable_to_prob_density_function(x = X_d)

    # determining prob. density of Y
    Y_d = local_dataset[:, Y_indices]
    p_Y_d = convert_variable_to_prob_density_function(x = Y_d)

    # determining prob. density of X,Y
    p_X_d_Y_d = convert_variable_to_prob_density_function(x = np.hstack((X_d,Y_d)))


    # summing over all density functions
    for X_d_key, p_X_d_value in p_X_d.items():
        for Y_d_key, p_Y_d_value in p_Y_d.items():
            p_X_d_Y_d_value = p_X_d_Y_d.get(X_d_key + Y_d_key, 0)
            dep += abs(p_X_d_Y_d_value - p_X_d_value * p_Y_d_value)


    return(dep)


# %%
def bp_dependency(dataset, X_indices, Y_indices, binning_indices= None, binning_strategy = 'auto', midway_binning = False, format_input = True) -> float:

    """
    This function determines the Berkelmans-Pries dependency of Y on X (notation: Dep(Y|X)) for a given dataset.

    Args:
    --------
        dataset (array_like): MxK array containing M samples of K variables.
        X_indices (array_like): 1-dimensional list /numpy.ndarray containing the indices for the X variable.
        Y_indices (array_like): 1-dimensional list / numpy.ndarray containing the indices for the Y variable.
        binning_indices (array_like, optional): 1-dimensional list / numpy.ndarray containing the indices that need to be binned. Default is `None`, which means that no variables are binned.
        binning_strategy (dictionary or number or str, optional): Default is `auto`. See numpy.histogram_bin_edges. Input a dictionary if for each binning index a specific strategy should be applied.
        midway_binning (bool, optional): Determines if the dataset is binned using the index of the bin (False) or the midway of the bin (True). Default is False.
        format_input (bool, optional): Default is True. If False, no additional checks are done for the input.


    Returns:
    --------
        float: The Berkelmans-Pries dependency score of Y on X. If Y is constant, np.NaN is returned.

    Raises:
    --------


    See also:
    --------
        unordered_bp_dependency, bin_data, numpy.histogram_bin_edges

    Example:
    --------
        >>> X_indices, Y_indices, dataset = (np.array([0]), np.array([1]), np.array([[0,0], [1,1], [0,2],[1,3]]))
        >>> print(bp_dependency(dataset= dataset, X_indices= X_indices, Y_indices= Y_indices))
        0.6666666666666666
    """

    if is_Y_constant(Y_indices, dataset) == True:
        return(np.NaN)

    numerator_result = unordered_bp_dependency(dataset= dataset, X_indices= X_indices, Y_indices= Y_indices, binning_indices= binning_indices, binning_strategy = binning_strategy, midway_binning = midway_binning, format_input = format_input)

    denominator_result = unordered_bp_dependency(dataset= dataset, X_indices= Y_indices, Y_indices= Y_indices, binning_indices= binning_indices, binning_strategy = binning_strategy, midway_binning = midway_binning, format_input = format_input)


    if denominator_result == 0.0:
        return(np.NaN)
    else:
        return(numerator_result / denominator_result)


# %%
