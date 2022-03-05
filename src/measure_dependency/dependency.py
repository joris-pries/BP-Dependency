# %%
from email.policy import default
import numpy as np
import scipy.integrate as sci
import scipy.stats as scs
from collections import defaultdict
import pandas as pd
import sympy as sp

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
        x (array_like): 1-dimensional list/numpy.ndarray containing the values that need to be binned.
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

# # Function used to kde
# def kde_data(dataset, indices, dataset_name, bw_method= None, overwrite = False):
#     """
#     This function determines the kernel density estimation using `scipy.stats.gaussian_kde` and saves the results in dic_kde.
#     """

#     global dic_kde
#     if not 'dic_kde' in globals():
#         dic_kde = {}

#     indices = np.asarray(indices)
#     hash_name = dataset_name + np.array2string(indices)

#     # If the results need te be overwritten or there is yet no result:
#     if overwrite == True or (hash_name not in dic_kde):
#         if dataset.ndim == 1:
#             x = dataset[:]
#         else:
#             x = dataset[:, indices]
#         x = np.squeeze(x).T
#         f_x = scs.gaussian_kde(x, bw_method= bw_method)
#         dic_kde[hash_name] = f_x

#     # If the saved results can be used:
#     elif overwrite == False and (hash_name in dic_kde):
#         f_x = dic_kde[hash_name]
#     return(f_x)
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
def unordered_bp_dependency(dataset, X_indices, Y_indices, binning_indices= None, binning_strategy = 'auto', midway_binning = False, return_binned_dataset = False, format_input = True) -> float:

    """
    This function is used for the `dependency' function and determines the unordered Berkelmans-Pries dependency of Y and X (notation: UD(Y,X)) for a given dataset.

    Args:
    --------
        dataset (array_like): MxK array containing M samples of K variables.
        X_indices (array_like): 1-dimensional list /numpy.ndarray containing the indices for the X variable.
        Y_indices (array_like): 1-dimensional list / numpy.ndarray containing the indices for the Y variable.
        binning_indices (array_like, optional): 1-dimensional list / numpy.ndarray containing the indices that need to be binned. Default is `None`, which means that no variables are binned.
        binning_strategy (dictionary or number or str): Default is `auto`. See numpy.histogram_bin_edges. Dictionary if for each binning index a specific strategy should be applied.
        kde_strategy (dictionary or number or str): Default is `None`. See scipy.stats.gaussian_kde. Dictionary if for each continuous variable a specific bandwidth should be applied.
        overwrite (bool): Default is `False`. Determines if saved kde and binning is used.
        dataset_name (str): Default is `default`. Used for saving and obtaining saved kde and binning

    Returns:
    --------
        float: The unordered Berkelmans-Pries dependency score of Y and X

    Raises:
    --------
        NotImplementedError
            If there is a mix of discrete and continuous variables.

    See also:
    --------
        bp_dependency

    Example:
    --------
        >>> test
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

    #TODO: opzoeken of dit ook al voor de UD variant geldt
    # if Y is constant, the dependency is by definition -1. Dit heeft ook gevolg voor de code in bp_dependency
    if is_Y_constant(Y_indices= Y_indices, dataset= local_dataset):
        if return_binned_dataset:
            return({'binned_dataset': local_dataset, 'unordered_bp_dependency' : -1.0})
        elif not return_binned_dataset:
            return(-1.0)


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

    if return_binned_dataset:
        return({'binned_dataset': local_dataset, 'unordered_bp_dependency' : dep})
    elif not return_binned_dataset:
        return(dep)


# %%
def bp_dependency(dataset, X_indices, Y_indices, binning_indices= None, binning_strategy = 'auto', midway_binning = False, return_binned_dataset = False, format_input = True) -> float:


    numerator_result = unordered_bp_dependency(dataset= dataset, X_indices= X_indices, Y_indices= Y_indices, binning_indices= binning_indices, binning_strategy = binning_strategy, midway_binning = midway_binning, return_binned_dataset = return_binned_dataset, format_input = format_input)

    denominator_result = unordered_bp_dependency(dataset= dataset, X_indices= Y_indices, Y_indices= Y_indices, binning_indices= binning_indices, binning_strategy = binning_strategy, midway_binning = midway_binning, return_binned_dataset = return_binned_dataset, format_input = format_input)

    if return_binned_dataset:
        if numerator_result['unordered_bp_dependency'] == -1.0 or denominator_result['unordered_bp_dependency'] == -1.0:
            return({'binned_dataset': numerator_result['binned_dataset'], 'bp_dependency' : -1.0})
        else:
            return({'binned_dataset': numerator_result['binned_dataset'], 'bp_dependency' : numerator_result['unordered_bp_dependency'] / denominator_result['unordered_bp_dependency']})

    elif not return_binned_dataset:
        if numerator_result == -1.0 or denominator_result == -1.0:
            return(-1.0)
        else:
            return(numerator_result / denominator_result)


# %%

# def unordered_bp_dependency(dataset, X_discrete_indices, X_continuous_indices, Y_discrete_indices, Y_continuous_indices, binning_strategy = 'auto', kde_strategy = None, overwrite = False, dataset_name= 'default') -> float:
#     """
#     This function is used for the `dependency' function and determines the unordered Berkelmans-Pries dependency of Y and X (notation: UD(Y,X)) for a given dataset. It is assumed that the discrete variables are already binned.

#     Args:
#     --------
#         dataset (numpy.ndarray): MxK array containing M samples of K variables.
#         X_discrete_indices (array_like): 1-dimensional list/numpy.ndarray containing the discrete indices for the X variable.
#         X_continous_indices (array_like): 1-dimensional list/numpy.ndarray containing the continous indices for the X variable.
#         Y_discrete_indices (array_like): 1-dimensional list/numpy.ndarray containing the discrete indices for the Y variable.
#         Y_continous_indices (array_like): 1-dimensional list/numpy.ndarray containing the continous indices for the Y variable.
#         binning_strategy (dictionary or number or str): Default is `auto`. See numpy.histogram_bin_edges. Dictionary if for each binning index a specific strategy should be applied.
#         kde_strategy (dictionary or number or str): Default is `None`. See scipy.stats.gaussian_kde. Dictionary if for each continuous variable a specific bandwidth should be applied.
#         overwrite (bool): Default is `False`. Determines if saved kde and binning is used.
#         dataset_name (str): Default is `default`. Used for saving and obtaining saved kde and binning

#     Returns:
#     --------
#         float: The unordered Berkelmans-Pries dependency score of Y and X

#     Raises:
#     --------
#         NotImplementedError
#             If there is a mix of discrete and continuous variables.

#     See also:
#     --------
#         bp_dependency

#     Example:
#     --------
#         >>> test
#     """

#     # copy dataset to local dataset
#     local_dataset = dataset.copy()

#     # initialization dep variable
#     dep = 0


#     if (X_discrete_indices is not None):
#         X_d = local_dataset[:, X_discrete_indices]
#         p_X_d = convert_variable_to_prob_density_function(x = X_d)

#     if (Y_discrete_indices is not None):
#         Y_d = local_dataset[:, Y_discrete_indices]
#         p_Y_d = convert_variable_to_prob_density_function(x = Y_d)

#     if (X_discrete_indices is not None) and (Y_discrete_indices is not None):
#         p_X_d_Y_d = convert_variable_to_prob_density_function(x = np.hstack((X_d,Y_d)))

#     if (X_continuous_indices is not None):
#         X_c = local_dataset[:, X_continuous_indices]
#         #TODO: kde strategy different for each index. Ik heb nu specifiek de eerste gekozen, omdat je anders unhashable type errors krijgt
#         print(tuple(X_continuous_indices.tolist()))
#         print(kde_strategy[tuple(X_continuous_indices.tolist())])
#         f_X_c = kde_data(x = X_c, bw_method= kde_strategy[tuple(X_continuous_indices.tolist())])

#     if (Y_continuous_indices is not None):
#         Y_c = local_dataset[:, Y_continuous_indices]
#         #TODO: kde strategy different for each index. Ik heb nu specifiek de eerste gekozen, omdat je anders unhashable type errors krijgt
#         f_Y_c = kde_data(x = Y_c, bw_method= kde_strategy[tuple(Y_continuous_indices.tolist())])

#     if (X_continuous_indices is not None) and (Y_continuous_indices is not None):
#         a = 1
#         #TODO: kde strategy different for each index. Ik heb nu specifiek de eerste gekozen, omdat je anders unhashable type errors krijgt
#         f_X_c_Y_c = kde_data(x = np.hstack((X_c,Y_c)), bw_method= kde_strategy[(X_continuous_indices + Y_continuous_indices)[0]])


#     # If all variables are discrete
#     if (X_continuous_indices is None) and (Y_continuous_indices is None):
#         for X_d_key, p_X_d_value in p_X_d.items():
#             for Y_d_key, p_Y_d_value in p_Y_d.items():
#                 p_X_d_Y_d_value = p_X_d_Y_d.get(X_d_key + Y_d_key, 0)
#                 dep += abs(p_X_d_Y_d_value - p_X_d_value * p_Y_d_value)
#         return(dep)

#     # If all variables are continuous
#     if (X_discrete_indices is None) and (Y_discrete_indices is None):
#         X_c_mins = np.amin(X_c, axis = 0)
#         X_c_maxs = np.amax(X_c, axis = 0)
#         Y_c_mins = np.amin(Y_c, axis = 0)
#         Y_c_maxs = np.amax(Y_c, axis = 0)

#         # Add bandwidth to either side of the integration range
#         X_c_integration_ranges = [[X_c_mins[i] - kde_strategy[X_continuous_indices[i]], X_c_maxs[i] + kde_strategy[X_continuous_indices[i]]] for i, red in enumerate(X_continuous_indices)]
#         Y_c_integration_ranges = [[Y_c_mins[i] - kde_strategy[Y_continuous_indices[i]], Y_c_maxs[i] + kde_strategy[Y_continuous_indices[i]]] for i, red in enumerate(Y_continuous_indices)]


#         def f_dep(*args):
#             len_X_continuous_indices = len(X_continuous_indices)
#             len_Y_continuous_indices = len(Y_continuous_indices)
#             x = args[:len_X_continuous_indices]
#             y = args[-len_Y_continuous_indices:]
#             return(abs(f_X_c_Y_c([*args]) - f_X_c(x) * f_Y_c(y)))

#         #options = {'limit': 100}
#         # dep = sci.nquad(f_dep, np.vstack((X_c_integration_ranges, Y_c_integration_ranges)), opts = [options, options])
#         dep = sci.nquad(f_dep, np.vstack((X_c_integration_ranges, Y_c_integration_ranges)))[0]
#         return(dep)

#     # If there is a mix between continuous and discrete variables
#     if ((X_discrete_indices is not None) and (X_continuous_indices is not None)) or ((Y_discrete_indices is not None) and (Y_continuous_indices is not None)):
#         raise NotImplementedError('This functionality has not yet been programmed. Instead of a mix of discrete and continuous variables, make them all discrete or all continuous.')
#     return None



# def bp_dependency(Y_indices, X_indices, dataset, kde_indices= None, binning_indices= None, binning_strategy = 'auto', kde_strategy = None) -> float:
#     """
#     This function determines the Berkelmans-Pries dependency of Y on X (notation: Dep(Y|X)) for a given dataset. Continuous variables need to be estimated using kernel density estimation or discretized using data binning.

#     Args:
#     --------
#         Y_indices (array_like): 1-dimensional list/numpy.ndarray containing the indices for the Y variable.
#         X_indices (array_like): 1-dimensional list/numpy.ndarray containing the indices for the X variable.
#         dataset (numpy.ndarray): MxK array containing M samples of K variables.
#         kde_indices (array_like): Default is None. 1-dimensional list/numpy.ndarray containing the indices for kernel density estimation.
#         binning_indices (array_like): Default is None. 1-dimensional list/numpy.ndarray containing the indices for data binning.
#         binning_strategy (dictionary or number or str): Default is `auto`. See numpy.histogram_bin_edges. Dictionary if for each binning index a specific strategy should be applied.
#         kde_strategy (dictionary or number or str): Default is `None`. See scipy.stats.gaussian_kde. Dictionary if for each continuous variable a specific bandwidth should be applied.
#     Returns:
#     --------
#         float: The Berkelmans-Pries dependency score of the Y variables on the X variables

#     Raises:
#     --------
#         ValueError
#             If indices are outside the dimensions of the dataset.

#     See also:
#     --------

#     Example:
#     --------
#         >>> test
#     """

#     # Convert indices to numpy array
#     Y_indices = np.asarray(Y_indices)
#     X_indices = np.asarray(X_indices)
#     X_continuous_indices = None
#     X_discrete_indices = X_indices
#     Y_continuous_indices = None
#     Y_discrete_indices = Y_indices

#     if kde_indices is not None:
#         kde_indices = np.asarray(kde_indices)
#         X_continuous_indices = np.intersect1d(X_indices, kde_indices)
#         X_discrete_indices = np.setdiff1d(X_indices, kde_indices)
#         if len(X_discrete_indices) == 0:
#             X_discrete_indices = None
#         Y_continuous_indices = np.intersect1d(Y_indices, kde_indices)
#         Y_discrete_indices = np.setdiff1d(Y_indices, kde_indices)
#         if len(Y_discrete_indices) == 0:
#             Y_discrete_indices = None
#         if not isinstance(kde_strategy, dict):
#             #kde_strategy = dict.fromkeys(kde_indices, kde_strategy)
#             help_kde = kde_strategy
#             kde_strategy = defaultdict(lambda: help_kde)

#     if binning_indices is not None:
#         binning_indices = np.asarray(binning_indices)
#         if not isinstance(binning_strategy, dict):
#             #binning_strategy = dict.fromkeys(binning_indices, binning_strategy)
#             binning_strategy = defaultdict(lambda: binning_strategy)


#     # Determine dimensions of dataset
#     dimensions_dataset = dataset.shape
#     n_samples = dimensions_dataset[0]
#     n_variables = dimensions_dataset[1]

#     # Rais ValueError if indices are outside of the scope of the dataset
#     if (np.amax(Y_indices) >= n_variables) or (np.amin(Y_indices) < 0):
#         raise ValueError('Y_indices are outside the scope of the dataset')
#     if (np.amax(X_indices) >= n_variables) or (np.amin(X_indices) < 0):
#         raise ValueError('X_indices are outside the scope of the dataset')
#     if kde_indices is not None:
#         if (np.amax(kde_indices) >= n_variables) or (np.amin(kde_indices) < 0):
#             raise ValueError('kde_indices are outside the scope of the dataset')
#     if binning_indices is not None:
#         if (np.amax(binning_indices) >= n_variables) or (np.amin(binning_indices) < 0):
#             raise ValueError('binning_indices are outside the scope of the dataset')

#     # copy dataset to local dataset
#     local_dataset = dataset.copy()


#     # Binning all binning_indices
#     if binning_indices is not None:
#         for bin_index in binning_indices:
#             local_dataset[:, bin_index] = bin_data(x= local_dataset[:, bin_index], bins = binning_strategy[bin_index])

#     # if Y is constant, the dependency is by definition -1.
#     if is_Y_constant(Y_indices= Y_indices, dataset= local_dataset):
#         return(-1.0)





#     numerator = unordered_bp_dependency(dataset= local_dataset, X_discrete_indices= X_discrete_indices, X_continuous_indices= X_continuous_indices, Y_discrete_indices= Y_discrete_indices, Y_continuous_indices= Y_continuous_indices, kde_strategy= kde_strategy )

#     # if Y consists only of continuous variables, the denominator is always 2.
#     if Y_discrete_indices is None:
#         denominator = 2
#     else:
#         denominator = unordered_bp_dependency(dataset= local_dataset, X_discrete_indices= Y_discrete_indices, X_continuous_indices= Y_continuous_indices, Y_discrete_indices= Y_discrete_indices, Y_continuous_indices= Y_continuous_indices, kde_strategy= kde_strategy )

#     return(numerator / denominator)

# # def theoretical_dependency(f_Y, f_X, f_X_Y, x_lower= -sp.oo, x_upper= sp.oo, y_lower= -sp.oo, y_upper= sp.oo) -> float:
# #     """
# #     This function determines the theoretical dependency of Y on X (notation: Dep(Y|X)) for given density functions.

# #     Args:
# #     --------

# #     Returns:
# #     --------
# #         float: The theoretical dependency score of the Y variables on the X variables

# #     Raises:
# #     --------

# #     See also:
# #     --------

# #     Example:
# #     --------
# #         >>> test
# #     """

# #     x, y = sp.symbols('q1 q2')

# #     y_symbol = list(f_Y.free_symbols)[0]
# #     x_symbol = list(f_X.free_symbols)[0]

# #     f_Y = f_Y.subs(y_symbol, y)
# #     f_X = f_X.subs(x_symbol, x)
# #     f_X_Y = f_X_Y.subs({y_symbol: y, x_symbol: x})

# #     expr = sp.Abs(f_X_Y - f_X * f_Y)
# #     numerator = sp.simplify(sp.integrate(expr, (x, x_lower, x_upper), (y, y_lower, y_upper)))
# #     print(numerator)

# #     expr = sp.Pow(f_Y, 2)

# #     list_of_points = []
# #     diracdeltas = expr.atoms(sp.DiracDelta)
# #     for delta in diracdeltas:
# #         solve_expr = delta.args[0]
# #         zero_point = sp.solve(solve_expr)
# #         list_of_points += zero_point

# #     sum_denominator = 0
# #     # Loop through all unique points
# #     for point in set(list_of_points):
# #         if (point >= y_lower) and (point <= y_upper):
# #             sum_denominator += expr.subs(y, point)

# #     denominator = 2 * (1 - sum_denominator)

# #     return(sp.simplify((numerator / denominator).subs({x: x_symbol, y:y_symbol})))
# # %%
# # %load_ext snakeviz
# # %%
# # import cProfile
# # import pstats

# # dataset= np.random.uniform(size= (500,3))
# # Y_indices = [0]
# # X_indices = [2]
# # kde_indices= [0,2]
# # binning_indices= None
# # binning_strategy = 'auto'
# # kde_strategy = 0.2
# # with cProfile.Profile() as pr:
# #     dependency(Y_indices= Y_indices, X_indices= X_indices, dataset= dataset, kde_indices= kde_indices, binning_indices= binning_indices, binning_strategy= binning_strategy, kde_strategy= kde_strategy)

# # stats = pstats.Stats(pr)
# # stats.sort_stats(pstats.SortKey.TIME)
# # stats.print_stats()
# # %%
# # x, y = sp.symbols('x y')
# # f_Y = sp.Piecewise(
# #     (0, y < 0),
# #     (0, y > 1),
# #     (1, True)
# # )


# # f_X = sp.Piecewise(
# #     (0, x < 0),
# #     (x, x <= 1),
# #     (2-x, x <= 2),
# #     (0, True)
# # )

# # f_X_Y = sp.Piecewise(
# #     (0, y < 0),
# #     (0, y > 1),
# #     (0, x < 0),
# #     (0, x > 2),
# #     (0, x - y > 1),
# #     (0, x - y < 0),
# #     (1, True)
# # )
# # # %%
# # theoretical_dependency(f_X, f_Y, f_X_Y, x_lower= 0, x_upper= 18, y_lower= 0, y_upper= 198)
# # %%

# # %%

# # %%

# # %%
# # Y_indices = [0]
# # X_indices = [1,2]
# # dataset = np.random.uniform(size = (5,6))
# # binning_indices = [0]
# # binning_strategy = int(1)

# #  # %%
# # bp_dependency(Y_indices= Y_indices, X_indices= X_indices, dataset= dataset, binning_indices= binning_indices, binning_strategy= binning_strategy) == -1
# # %%
