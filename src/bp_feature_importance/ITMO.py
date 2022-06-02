# %%
from collections import Counter
from itertools import groupby
from math import log, fsum
from operator import itemgetter

import numpy as np


def conditional_entropy(x_j, y):
    """Calculate the conditional entropy (H(Y|X)) between two arrays.

    Parameters
    ----------
    x_j : array-like, shape (n,)
        The first array.
    y : array-like, shape (n,)
        The second array.

    Returns
    -------
    float : H(Y|X) value

    Examples
    --------
    >>> from ITMO_FS.utils.information_theory import conditional_entropy
    >>> conditional_entropy([1,2,1,3,4], [1,2,3,4,5])
    0.2772588722239781
    >>> conditional_entropy([1], [2])
    0.0
    >>> conditional_entropy([1,2,1,3,2,4], [3,3,3,3,3,3])
    0.0
    >>> conditional_entropy([1,2,3,1,3,2,3,4,1], [1,2,1,3,1,4,4,1,5])
    0.7324081924454064
    """
    buf = [[e[1] for e in g] for _, g in 
           groupby(sorted(zip(x_j, y)), itemgetter(0))]
    return fsum(entropy(group) * len(group) for group in buf) / len(x_j)


def joint_entropy(*arrs):
    """Calculate the joint entropy (H(X;Y;...)) between multiple arrays.

    Parameters
    ----------
    arrs : any number of array-like, all of shape (n,)
        Any number of arrays.

    Returns
    -------
    float : H(X;Y;...) value

    Examples
    --------
    >>> from ITMO_FS.utils.information_theory import joint_entropy
    >>> joint_entropy([1,2,3,4,5])
    1.6094379124341003
    >>> joint_entropy([1,2,3,4,4])
    1.3321790402101221
    >>> joint_entropy([1,2,3,4,5,6], [1,2,3,4,5,6], [1,2,3,4,5,6])
    1.791759469228055
    >>> joint_entropy([1,2,1,3,2], [3,3,3,3,3])
    1.0549201679861442
    >>> conditional_entropy([1,1], [2,2])
    0.0
    """
    return entropy(list(zip(*arrs)))


def matrix_mutual_information(x, y):
    """Calculate the mutual information (I(X;Y) = H(Y) - H(Y|X)) between each
    column of the matrix and an array.

    Parameters
    ----------
    x : array-like, shape (n, n_features)
        The matrix.
    y : array-like, shape (n,)
        The second array.

    Returns
    -------
    array-like, shape (n_features,) : I(X;Y) values for all columns of the
    matrix

    Examples
    --------
    >>> from ITMO_FS.utils.information_theory import matrix_mutual_information
    >>> matrix_mutual_information([[1,3,2,1], [2,2,2,1], [3,3,2,2]], [1,1,2])
    array([0.63651417, 0.17441605, 0.        , 0.63651417])
    """
    return np.apply_along_axis(mutual_information, 0, x, y)


def mutual_information(x, y):
    """Calculate the mutual information (I(X;Y) = H(Y) - H(Y|X)) between two
    arrays.

    Parameters
    ----------
    x : array-like, shape (n,)
        The first array.
    y : array-like, shape (n,)
        The second array.

    Returns
    -------
    float : I(X;Y) value

    Examples
    --------
    >>> from ITMO_FS.utils.information_theory import mutual_information
    >>> mutual_information([1,2,3,4,5], [5,4,3,2,1])
    1.6094379124341003
    >>> mutual_information([1,2,3,1,2,3,1,2,3], [1,1,2,2,3,3,4,4,5])
    0.48248146150371407
    >>> mutual_information([1,2,3], [1,1,1])
    0.0
    >>> mutual_information([1,2,1,3,2,4,3,1], [1,2,3,4,2,3,2,1])
    0.9089087348987808
    """
    return entropy(y) - conditional_entropy(x, y)


def conditional_mutual_information(x, y, z):
    """Calculate the conditional mutual information (I(X;Y|Z) = H(X;Z) + H(Y;Z)
    - H(X;Y;Z) - H(Z)) between three arrays.

    Parameters
    ----------
    x : array-like, shape (n,)
        The first array.
    y : array-like, shape (n,)
        The second array.
    z : array-like, shape (n,)
        The third array.

    Returns
    -------
    float : I(X;Y|Z) value

    Examples
    --------
    >>> from ITMO_FS.utils import conditional_mutual_information
    >>> conditional_mutual_information([1,3,2,1], [2,2,2,1], [3,3,2,2])
    0.3465735902799726
    >>> conditional_mutual_information([1,1,1,1,1], [2,3,4,2,1], [1,2,1,2,1])
    0.0
    >>> conditional_mutual_information([1,2,3,4,1], [2,3,4,2,1], [1,1,1,1,1])
    1.054920167986144
    >>> conditional_mutual_information([1,2,3], [1,1,1], [3,2,2])
    0.0
    >>> conditional_mutual_information([1,2,3,4,1,3,2,1,4,5],
    ... [1,3,2,4,5,4,3,2,1,2], [2,1,4,3,2,6,5,2,1,3])
    0.27725887222397816
    """
    return (entropy(list(zip(x, z)))
            + entropy(list(zip(y, z)))
            - entropy(list(zip(x, y, z)))
            - entropy(z))


def joint_mutual_information(x, y, z):
    """Calculate the joint mutual information (I(X,Y;Z) = I(X;Z) + I(Y;Z|X))
    between three arrays.

    Parameters
    ----------
    x : array-like, shape (n,)
        The first array.
    y : array-like, shape (n,)
        The second array.
    z : array-like, shape (n,)
        The third array.

    Returns
    -------
    float : I(X,Y;Z) value

    Examples
    --------
    >>> from ITMO_FS.utils.information_theory import joint_mutual_information
    >>> joint_mutual_information([1,3,2,1], [2,2,2,1], [3,3,2,2])
    0.6931471805599454
    >>> joint_mutual_information([1,1,1,1,1], [2,3,4,2,1], [1,2,1,2,1])
    0.39575279478527814
    >>> joint_mutual_information([1,2,3,4,1], [2,3,4,2,1], [1,1,1,1,1])
    0.0
    >>> joint_mutual_information([1,2,3], [1,1,1], [3,2,2])
    0.636514168294813
    >>> joint_mutual_information([1,2,3,4,1,3,2,1,4,5],
    ... [1,3,2,4,5,4,3,2,1,2], [2,1,4,3,2,6,5,2,1,3])
    1.5571130980576458
    """
    return mutual_information(x, z) + conditional_mutual_information(y, z, x)


def interaction_information(x, y, z):
    """Calculate the interaction information (I(X;Y;Z) = I(X;Y) - I(X;Y|Z))
    between three arrays.

    Parameters
    ----------
    x : array-like, shape (n,)
        The first array.
    y : array-like, shape (n,)
        The second array.
    z : array-like, shape (n,)
        The third array.

    Returns
    -------
    float : I(X;Y;Z) value

    Examples
    --------
    >>> from ITMO_FS.utils.information_theory import interaction_information
    >>> interaction_information([1,3,2,1], [2,2,2,1], [3,3,2,2])
    -0.13081203594113694
    >>> interaction_information([1,1,1,1,1], [2,3,4,2,1], [1,2,1,2,1])
    0.0
    >>> interaction_information([1,2,3,4,1], [2,3,4,2,1], [1,1,1,1,1])
    0.0
    >>> interaction_information([1,2,3], [1,1,1], [3,2,2])
    0.0
    >>> interaction_information([1,2,3,4,1,3,2,1,4,5],
    ... [1,3,2,4,5,4,3,2,1,2], [2,1,4,3,2,6,5,2,1,3])
    0.6730116670092565
    """
    return mutual_information(x, y) - conditional_mutual_information(x, y, z)


def symmetrical_relevance(x, y, z):
    """Calculate the symmetrical relevance (SR(X;Y;Z) = I(X;Y;Z) / H(X;Y|Z))
    between three arrays.

    Parameters
    ----------
    x : array-like, shape (n,)
        The first array.
    y : array-like, shape (n,)
        The second array.
    z : array-like, shape (n,)
        The third array.

    Returns
    -------
    float : SR(X;Y;Z) value

    Examples
    --------
    >>> from ITMO_FS.utils.information_theory import symmetrical_relevance
    >>> symmetrical_relevance([1,3,2,1], [2,2,2,1], [3,3,2,2])
    0.5000000000000001
    >>> symmetrical_relevance([1,1,1,1,1], [2,3,4,2,1], [1,2,1,2,1])
    0.2458950368496943
    >>> symmetrical_relevance([1,2,3,4,1], [2,3,4,2,1], [1,1,1,1,1])
    0.0
    >>> symmetrical_relevance([1,2,3], [1,1,1], [3,2,2])
    0.5793801642856952
    >>> symmetrical_relevance([1,2,3,4,1,3,2,1,4,5],
    ... [1,3,2,4,5,4,3,2,1,2], [2,1,4,3,2,6,5,2,1,3])
    0.6762456261857126
    """
    return joint_mutual_information(x, y, z) / joint_entropy(x, y, z)

def entropy(x):
    """Calculate the entropy (H(X)) of an array.

    Parameters
    ----------
    x : array-like, shape (n,)
        The array.

    Returns
    -------
    float : H(X) value

    Examples
    --------
    >>> from ITMO_FS.utils.information_theory import entropy
    >>> entropy([1,1,1])
    0.0
    >>> entropy([1,2,3,4,5])
    1.6094379124341003
    >>> entropy([5,4,1,2,3])
    1.6094379124341003
    >>> entropy([1,2,1,2,1,2,1,2,1,2])
    0.6931471805599456
    >>> entropy([1,1,1,1,1,2])
    0.4505612088663047
    """
    return log(len(x)) - fsum(v * log(v) for v in Counter(x).values()) / len(x)


import math
from functools import partial

import numpy as np
from qpsolvers import solve_qp
from scipy.linalg import sqrtm


def qpfs_body(X, y, fn, alpha=None, r=None, sigma=None, solv='quadprog',
              metric_for_complex=complex.__abs__):
    # TODO understand why complex double appears
    # TODO find suitable r parameter value
    # TODO find suitable sigma parameter value
    if r is None:
        r = X.shape[1] - 1
    if r >= X.shape[1]:
        raise TypeError("r parameter should be less than the number of features")
    F = np.zeros(X.shape[1], dtype=np.double)  # F vector represents how each variable is correlated class
    class_size = max(
        y) + 1  # Count the number of classes, we assume that class labels would be numbers from 1 to max(y)
    priors = np.histogram(y, bins=max(y))[0]  # Count prior probabilities of classes
    for i in range(1, class_size):  # Loop through classes
        Ck = np.where(y == i, 1, 0)  # Get array C(i) where C(k) is 1 when i = k and 0 otherwise
        F += priors[i - 1] * fn(X, Ck)  # Counting F vector
    Q = np.apply_along_axis(partial(fn, X), 0, X).reshape(X.shape[1], X.shape[1])
    indices = np.random.random_integers(0, Q.shape[0] - 1,
                                        r)  # Taking random r indices according to Nystrom approximation
    A = Q[indices][:, :r]  # A matrix for Nystrom(matrix of real numbers with size of [r, r])
    B = Q[indices][:, r:]  # B matrix for Nystrom(matrix of real numbers with size of [r, M - r])
    if alpha is None:
        alpha = __countAlpha(A, B, F)  # Only in filter method, in wrapper we should adapt it based on performance
    AInvSqrt = sqrtm(np.linalg.pinv(A))  # Calculate squared root of inverted matrix A
    S = np.add(A, AInvSqrt.dot(B).dot(B.T).dot(AInvSqrt))  # Caluclate S matrix
    eigvals, EVect = np.linalg.eig(S)  # eigenvalues and eigenvectors of S
    U = np.append(A, B.T, axis=0).dot(AInvSqrt).dot(EVect).dot(
        sqrtm(np.linalg.pinv(EVect)))  # Eigenvectors of Q matrix using [A B]
    eigvalsFilt, UFilt = __filterBy(sigma, eigvals,
                                    U)  # Take onyl eigenvalues greater than threshold and corresponding eigenvectors
    LFilt = np.zeros((len(eigvalsFilt), len(eigvalsFilt)), dtype=complex)  # initialize diagonal matrix of eigenvalues
    for i in range(len(eigvalsFilt)):  # Loop through eigenvalues
        LFilt[i][i] = eigvalsFilt[i]  # Init diagonal values
    UFilt = np.array([list(map(metric_for_complex, t)) for t in UFilt])
    LFilt = np.array([list(map(metric_for_complex, t)) for t in LFilt])
    yf = solve_qp((1 - alpha) * LFilt, alpha * F.dot(UFilt), UFilt, np.zeros(UFilt.shape[0]),
                  solver=solv)  # perform qp on stated problem
    xSolution = UFilt.dot(yf)  # Find x - weights of features
    forRanks = list(zip(xSolution, F, [x for x in range(len(F))]))  # Zip into array of tuple for proper sort
    forRanks.sort(reverse=True)
    ranks = np.zeros(len(F))
    rankIndex = 1
    for i in forRanks:
        ranks[int(i[2])] = rankIndex
        rankIndex += 1
    return ranks


def __filterBy(sigma, eigvals, U):
    if sigma is None:
        return eigvals, U
    y = np.where(eigvals > sigma)[0]
    return eigvals[y], U[:, y]


def __countAlpha(A, B, F):
    Comb = B.T.dot(np.linalg.pinv(A)).dot(B)
    sumQ = np.sum(A) + 2 * np.sum(B) + np.sum(Comb)
    sumQ /= (A.shape[1] + B.shape[1]) ** 2
    sumF = np.sum(F)
    sumF /= len(F)
    return sumQ / (sumQ + sumF)

from abc import abstractmethod
from logging import getLogger

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted


class BaseTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):
        """Fit the algorithm.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,), optional
            The class labels.
        fit_params : dict, optional
            Additional parameters to pass to underlying _fit function.

        Returns
        -------
        Self, i.e. the transformer object.
        """
        if y is not None:
            X, y = check_X_y(X, y, dtype='numeric')
            if y.dtype.kind == 'O':
                y = y.astype('int')
        else:
            X = check_array(X, dtype='float64', accept_large_sparse=False)

        self.n_total_features_ = X.shape[1]
        nonconst_features = VarianceThreshold().fit(X).get_support(indices=True)
        self.n_features_ = nonconst_features.shape[0]

        if self.n_features_ != self.n_total_features_:
            getLogger(__name__).warning(
                "Found %d constant features; they would not be used in fit")

        if hasattr(self, 'n_features'):
            if self.n_features > self.n_features_:
                getLogger(__name__).error(
                    "Cannot select %d features with n_features = %d",
                    self.n_features, self.n_features_)
                raise ValueError(
                    "Cannot select %d features with n_features = %d"
                    % (self.n_features, self.n_features_))

        if hasattr(self, 'epsilon'):
            if self.epsilon <= 0:
                getLogger(__name__).error(
                    "Epsilon should be positive, %d passed", self.epsilon)
                raise ValueError(
                    "Epsilon should be positive, %d passed" % self.epsilon)


        self._fit(X[:, nonconst_features], y, **fit_params)

        if hasattr(self, 'feature_scores_'):
            scores = np.empty(self.n_total_features_)
            scores.fill(np.nan)
            scores[nonconst_features] = self.feature_scores_
            self.feature_scores_ = scores
        self.selected_features_ = nonconst_features[self.selected_features_]

        return self

    def transform(self, X):
        """
            Transform given data by slicing it with selected features.

            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                The training input samples.

            Returns
            ------
            Transformed 2D numpy array
        """
        check_is_fitted(self, 'selected_features_')
        X_ = check_array(X, dtype='numeric', accept_large_sparse=False)
        if X_.shape[1] != self.n_total_features_:
            getLogger(__name__).error(
                "Shape of input is different from what was seen in 'fit'")
            raise ValueError(
                "Shape of input is different from what was seen in 'fit'")
        if isinstance(X, pd.DataFrame):
            return X[X.columns[self.selected_features_]]
        else:
            return X_[:, self.selected_features_]

    @abstractmethod
    def _fit(self, X, y):
        pass

from logging import getLogger

from sklearn.base import clone
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

# from . import BaseTransformer

class BaseWrapper(BaseTransformer):
    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):
        """Fit the algorithm.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,), optional
            The class labels.
        fit_params : dict, optional
            Additional parameters to pass to underlying _fit function.

        Returns
        -------
        Self, i.e. the transformer object.
        """
        if not hasattr(self.estimator, 'fit'):
            getLogger(__name__).error(
                "estimator should be an estimator implementing "
                "'fit' method, %s was passed", self.estimator)
            raise TypeError(
                "estimator should be an estimator implementing "
                "'fit' method, %s was passed" % self.estimator)
        if not hasattr(self.estimator, 'predict'):
            getLogger(__name__).error(
                "estimator should be an estimator implementing "
                "'predict' method, %s was passed", self.estimator)
            raise TypeError(
                "estimator should be an estimator implementing "
                "'predict' method, %s was passed" % self.estimator)
        self._estimator = clone(self.estimator)

        return super().fit(X, y, **fit_params)

    def predict(self, X):
        """Predict class labels for the input data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        array-like, shape (n_samples,) : class labels
        """
        check_is_fitted(self, 'selected_features_')
        X_ = check_array(X, dtype='float64', accept_large_sparse=False)
        if X_.shape[1] != self.n_features_:
            getLogger(__name__).error(
                "Shape of input is different from what was seen in 'fit'")
            raise ValueError(
                "Shape of input is different from what was seen in 'fit'")

        return self._estimator.predict(X_[:, self.selected_features_])

from numpy import array


def generate_features(X, features=None):
    if features is None:
        try:
            if X.columns is list:
                features = X.columns
            else:
                features = list(X.columns)
        except AttributeError:
            features = [i for i in range(X.shape[1])]
    return array(features)


def check_filters(filters):
    for filter_ in filters:
        attr = None
        if not hasattr(filter_, 'fit'):
            attr = 'fit'
        if not hasattr(filter_, 'transform'):
            attr = 'transform'
        if not hasattr(filter_, 'fit_transform'):
            attr = 'fit_transform'
        if not (attr is None):
            raise TypeError(
                "filters should be a list of filters each implementing {0} "
                "method, {1} was passed".format(attr, filter_))


def check_cutting_rule(cutting_rule):
    pass  # todo check cutting rule


RESTRICTIONS = {'qpfs_filter': {'__select_k'}}


def check_restrictions(measure_name, cutting_rule_name):
    if (measure_name in RESTRICTIONS.keys() and
            cutting_rule_name not in RESTRICTIONS[measure_name]):
        raise KeyError(
            "This measure %s doesn't support this cutting rule %s"
            % (measure_name, cutting_rule_name))

import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import euclidean_distances


def cartesian(rw, cl):  # returns cartesian product for passed numpy arrays as two paired numpy array
    tmp = np.array(np.meshgrid(rw, cl)).T.reshape(len(rw) * len(cl), 2)
    return tmp.T[0], tmp.T[1]

def weight_func(model):  # weight function used in MOS testing
    return model.coef_[0]

def f1_scorer(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')

def augmented_rvalue(X, y, k=7, theta=3):
    """Calculate the augmented R-value for a dataset with two classes.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input samples.
    y : array-like, shape (n_samples,)
        The classes for the samples.
    k : int
        The amount of nearest neighbors used in the calculation.
    theta : int
        The threshold value: if from k nearest neighbors of an object more than
        theta of them are of a different class, then this object is in the
        overlap region.

    Returns
    -------
    float - the augmented R-value for the dataset; the value is in the range
    [-1, 1].

    Notes
    -----
    For more details see `this paper <https://www.sciencedirect.com/science/article/pii/S0169743919306070>`_.
    """
    unique, counts = np.unique(y, return_counts=True)
    freq = sorted(list(zip(unique, counts)), key=lambda x: x[1], reverse=True)
    dm = euclidean_distances(X, X)
    Rs = []
    Cs = []

    for label, frequency in freq:
        Cs.append(frequency)
        count = 0
        for elem in [i for i, x in enumerate(y) if x == label]:
            nearest = knn_from_class(dm, y, elem, k, 1, anyClass=True)
            count += np.sign(
                k
                - list(map(lambda x: y[x], nearest)).count(label)
                - theta)
        Rs.append(count / frequency)
    Cs = Cs[::-1]
    return np.dot(Rs, Cs) / len(X)


def knn_from_class(distances, y, index, k, cl, anyOtherClass=False,
                   anyClass=False):
    """Return the indices of k nearest neighbors of X[index] from the selected
    class.

    Parameters
    ----------
    distances : array-like, shape (n_samples, n_samples)
        The distance matrix of the input samples.
    y : array-like, shape (n_samples,)
        The classes for the samples.
    index : int
        The index of an element.
    k : int
        The amount of nearest neighbors to return.
    cl : int
        The class label for the nearest neighbors.
    anyClass : bool
        If True, returns neighbors not belonging to the same class as X[index].

    Returns
    -------
    array-like, shape (k,) - the indices of the nearest neighbors
    """
    y_c = np.copy(y)
    if anyOtherClass:
        cl = y_c[index] + 1
        y_c[y_c != y_c[index]] = cl
    if anyClass:
        y_c.fill(cl)
    class_indices = np.nonzero(y_c == cl)[0]
    distances_class = distances[index][class_indices]
    nearest = np.argsort(distances_class)
    if y_c[index] == cl:
        nearest = nearest[1:]

    return class_indices[nearest[:k]]

def matrix_norm(M):
    """Calculate the norm of all rows in the matrix.

    Parameters
    ----------
    M : array-like, shape (n, m)
        The matrix.

    Returns
    -------
    array-like, shape (n,) : the norms for each row in the matrix
    """
    return np.sqrt((M * M).sum(axis=1))

def l21_norm(M):
    """Calculate the L2,1 norm of a matrix.

    Parameters
    ----------
    M : array-like, shape (n, m)
        The matrix.

    Returns
    -------
    float : the L2,1 norm of this matrix
    """
    return matrix_norm(M).sum()

def power_neg_half(M):
    """Calculate M ^ (-1/2).

    Parameters
    ----------
    M : array-like, shape (n, m)
        The matrix.

    Returns
    -------
    array-like, shape (n, m) : M ^ (-1/2)
    """
    return np.sqrt(np.linalg.inv(M))

def apply_cr(cutting_rule):
    """Extract the cutting rule from a tuple or callable.

    Parameters
    ----------
    cutting_rule : tuple or callable
        A (str, float) tuple describing a cutting rule or a callable with
        signature cutting_rule (features) which should return a list of features
        ranked by some rule.

    Returns
    -------
    callable : a cutting rule callable
    """
    # from ..filters.univariate.measures import CR_NAMES, MEASURE_NAMES
    if type(cutting_rule) is tuple:
        cutting_rule_name = cutting_rule[0]
        cutting_rule_value = cutting_rule[1]
        try:
            cr = CR_NAMES[cutting_rule_name](cutting_rule_value)
        except KeyError:
            raise KeyError("No %s cutting rule yet" % cutting_rule_name)
    elif hasattr(cutting_rule, '__call__'):
        cr = cutting_rule
    else:
        raise KeyError(
            "%s isn't a cutting rule function or string" % cutting_rule)
    return cr







############################################################################
from logging import getLogger

import numpy as np
from sklearn.preprocessing import OneHotEncoder

# from ...utils import l21_norm, matrix_norm, BaseTransformer


class RFS(BaseTransformer):
    """Robust Feature Selection via Joint L2,1-Norms Minimization algorithm.

    Parameters
    ----------
    n_features : int
        Number of features to select.
    gamma : float
        Regularization parameter.
    max_iterations : int
        Maximum amount of iterations to perform.
    epsilon : positive float
        Specifies the needed residual between the target functions from
        consecutive iterations. If the residual is smaller than epsilon, the
        algorithm is considered to have converged.

    Notes
    -----
    For more details see `this paper
    <https://papers.nips.cc/paper/3988-efficient-and-robust-feature-selection-via-joint-l21-norms-minimization.pdf/>`_.

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import RFS
    >>> import numpy as np
    >>> X = np.array([[1, 2, 3, 3, 1], [2, 2, 3, 3, 2], [1, 3, 3, 1, 3],
    ... [1, 1, 3, 1, 4], [2, 4, 3, 1, 5]])
    >>> y = np.array([1, 2, 1, 1, 2])
    >>> model = RFS(2).fit(X, y)
    >>> model.selected_features_
    array([0, 3], dtype=int64)
    """
    def __init__(self, n_features, gamma=1, max_iterations=1000, epsilon=1e-5):
        self.n_features = n_features
        self.gamma = gamma
        self.max_iterations = max_iterations
        self.epsilon = epsilon

    def _fit(self, X, y):
        """Fit the filter.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_classes)
            The target values or their one-hot encoding.

        Returns
        -------
        None
        """
        if len(y.shape) == 2:
            Y = y
        else:
            Y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()

        getLogger(__name__).info("Transformed Y: %s", Y)
        n_samples = X.shape[0]
        A = np.append(X, self.gamma * np.eye(n_samples), axis=1)
        getLogger(__name__).info("A: %s", A)
        D = np.eye(n_samples + self.n_features_)

        previous_target = -1
        for _ in range(self.max_iterations):
            D_inv = np.linalg.inv(D)
            U = D_inv.dot(A.T).dot(np.linalg.inv(A.dot(D_inv).dot(A.T))).dot(Y)
            getLogger(__name__).info("U: %s", U)
            diag = 2 * matrix_norm(U)
            diag[diag < 1e-10] = 1e-10  # prevents division by zero
            D = np.diag(1 / diag)
            getLogger(__name__).info("D: %s", D)

            target = l21_norm(U)
            getLogger(__name__).info("New target value: %d", target)
            if abs(target - previous_target) < self.epsilon:
                break
            previous_target = target

        getLogger(__name__).info("Ended up with U: %s", U)
        self.feature_scores_ = matrix_norm(U[:self.n_features_])
        getLogger(__name__).info("Feature scores: %s", self.feature_scores_)
        ranking = np.argsort(self.feature_scores_)[::-1]
        self.selected_features_ = ranking[:self.n_features]


from functools import partial, update_wrapper
from math import exp

import numpy as np
from scipy.sparse import lil_matrix
from scipy.stats import rankdata
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import pairwise_distances, euclidean_distances
from sklearn.neighbors import NearestNeighbors

# from ...utils.information_theory import conditional_entropy
# from ...utils.information_theory import entropy
# from ...utils.qpfs_body import qpfs_body
# from ...utils.functions import knn_from_class


def _wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def fit_criterion_measure(x, y):
    """Calculate the FitCriterion score for features. Bigger values mean more
    important features.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.

    Returns
    -------
    array-like, shape (n_features,) : feature scores

    See Also
    --------
    https://core.ac.uk/download/pdf/191234514.pdf

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import fit_criterion_measure
    >>> import numpy as np
    >>> x = np.array([[1, 2, 4, 1, 1], [2, 2, 2, 1, 2], [3, 5, 1, 1, 4],
    ... [1, 1, 1, 1, 4], [2, 2, 2, 1, 5]])
    >>> y = np.array([1, 2, 3, 1, 2])
    >>> fit_criterion_measure(x, y)
    array([1. , 0.8, 0.8, 0.4, 0.6])
    """
    def count_hits(feature):
        splits = {cl: feature[y == cl] for cl in classes}
        means = {cl: np.mean(splits[cl]) for cl in classes}
        devs = {cl: np.var(splits[cl]) for cl in classes}
        distances = np.vectorize(
            lambda x_val: {cl: (
                abs(x_val - means[cl])
                / (devs[cl] + 1e-10)) for cl in classes})(feature)
        return np.sum(np.vectorize(lambda d: min(d, key=d.get))(distances) == y)

    classes = np.unique(y)
    return np.apply_along_axis(count_hits, 0, x) / x.shape[0]


def f_ratio_measure(x, y):
    """Calculate Fisher score for features. Bigger values mean more important
    features.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.

    Returns
    -------
    array-like, shape (n_features,) : feature scores

    See Also
    --------
    https://papers.nips.cc/paper/2909-laplacian-score-for-feature-selection.pdf

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import f_ratio_measure
    >>> import numpy as np
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1],
    ... [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
    >>> f_ratio_measure(x, y)
    array([0.6 , 0.2 , 1.  , 0.12, 5.4 ])
    """
    def __F_ratio(feature):
        splits = {cl: feature[y == cl] for cl in classes}
        mean_feature = np.mean(feature)
        inter_class = np.sum(
            np.vectorize(lambda cl: (
                counts_d[cl]
                * np.power(mean_feature - np.mean(splits[cl]), 2)))(classes))
        intra_class = np.sum(
            np.vectorize(lambda cl: (
                counts_d[cl]
                * np.var(splits[cl])))(classes))
        return inter_class / (intra_class + 1e-10)

    classes, counts = np.unique(y, return_counts=True)
    counts_d = {cl: counts[idx] for idx, cl in enumerate(classes)}
    return np.apply_along_axis(__F_ratio, 0, x)


def gini_index(x, y):
    """Calculate Gini index for features. Bigger values mean more important
    features. This measure works best with discrete features due to being based
    on information theory.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.

    Returns
    -------
    array-like, shape (n_features,) : feature scores

    See Also
    --------
    http://lkm.fri.uni-lj.si/xaigor/slo/clanki/ijcai95z.pdf

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import gini_index
    >>> from sklearn.preprocessing import KBinsDiscretizer
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1],
    ... [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
    >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
    >>> x = est.fit_transform(x)
    >>> gini_index(x, y)
    array([0.14      , 0.04      , 0.64      , 0.24      , 0.37333333])
    """
    def __gini(feature):
        values, counts = np.unique(feature, return_counts=True)
        counts_d = {val: counts[idx] for idx, val in enumerate(values)}
        total_sum = np.sum(
            np.vectorize(
                lambda val: (
                    np.sum(
                        np.square(
                            np.unique(
                                y[feature == val], return_counts=True)[1]))
                    / counts_d[val]))(values))
        return total_sum / x.shape[0] - prior_prob_squared_sum

    classes, counts = np.unique(y, return_counts=True)
    prior_prob_squared_sum = np.sum(np.square(counts / x.shape[0]))

    return np.apply_along_axis(__gini, 0, x)


def su_measure(x, y):
    """SU is a correlation measure between the features and the class
    calculated via formula SU(X,Y) = 2 * I(X|Y) / (H(X) + H(Y)). Bigger values
    mean more important features. This measure works best with discrete
    features due to being based on information theory.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.

    Returns
    -------
    array-like, shape (n_features,) : feature scores

    See Also
    --------
    https://pdfs.semanticscholar.org/9964/c7b42e6ab311f88e493b3fc552515e0c764a.pdf

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import su_measure
    >>> from sklearn.preprocessing import KBinsDiscretizer
    >>> import numpy as np
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1],
    ... [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
    >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
    >>> x = est.fit_transform(x)
    >>> su_measure(x, y)
    array([0.28694182, 0.13715115, 0.79187567, 0.47435099, 0.67126949])
    """
    def __SU(feature):
        entropy_x = entropy(feature)
        return (2 * (entropy_x - conditional_entropy(y, feature))
                  / (entropy_x + entropy_y))

    entropy_y = entropy(y)
    return np.apply_along_axis(__SU, 0, x)

# TODO CONCORDATION COEF

def kendall_corr(x, y):
    """Calculate Sample sign correlation (Kendall correlation) for each
    feature. Bigger absolute values mean more important features.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.

    Returns
    -------
    array-like, shape (n_features,) : feature scores

    See Also
    --------
    https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import kendall_corr
    >>> import numpy as np
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1],
    ... [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
    >>> kendall_corr(x, y)
    array([-0.1,  0.2, -0.4, -0.2,  0.2])
    """
    def __kendall_corr(feature):
        k_corr = 0.0
        for i in range(len(feature)):
            k_corr += np.sum(np.sign(feature[i] - feature[i + 1:])
                             * np.sign(y[i] - y[i + 1:]))
        return 2 * k_corr / (feature.shape[0] * (feature.shape[0] - 1))

    return np.apply_along_axis(__kendall_corr, 0, x)


def fechner_corr(x, y):
    """Calculate Sample sign correlation (Fechner correlation) for each
    feature. Bigger absolute values mean more important features.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.

    Returns
    -------
    array-like, shape (n_features,) : feature scores

    See Also
    --------

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import fechner_corr
    >>> import numpy as np
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1],
    ... [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
    >>> fechner_corr(x, y)
    array([-0.2,  0.2, -0.4, -0.2, -0.2])
    """
    y_dev = y - np.mean(y)
    x_dev = x - np.mean(x, axis=0)
    return np.sum(np.sign(x_dev.T * y_dev), axis=1) / x.shape[0]

def reliefF_measure(x, y, k_neighbors=1):
    """Calculate ReliefF measure for each feature. Bigger values mean more
    important features.

    Note:
    Only for complete x
    Rather than repeating the algorithm m(TODO Ask Nikita about user defined)
    times, implement it exhaustively (i.e. n times, once for each instance)
    for relatively small n (up to one thousand).

    Calculates spearman correlation for each feature.
    Spearman's correlation assesses monotonic relationships (whether linear or
    not). If there are no repeated data values, a perfect Spearman correlation
    of +1 or −1 occurs when each of the variables is a perfect monotone
    function of the other.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The input samples.
    y : array-like, shape (n_samples,)
        The classes for the samples.
    k_neighbors : int, optional
        The number of neighbors to consider when assigning feature importance
        scores. More neighbors results in more accurate scores but takes
        longer. Selection of k hits and misses is the basic difference to
        Relief and ensures greater robustness of the algorithm concerning noise.

    Returns
    -------
    array-like, shape (n_features,) : feature scores

    See Also
    --------
    R.J. Urbanowicz et al. Relief-based feature selection: Introduction and
    review. Journal of Biomedical Informatics 85 (2018) 189–203

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import reliefF_measure
    >>> import numpy as np
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1],
    ... [3, 1, 4, 3, 1], [3, 1, 2, 3, 1], [1, 2, 1, 4, 2], [4, 3, 2, 3, 1]])
    >>> y = np.array([1, 2, 2, 1, 2, 1, 2])
    >>> reliefF_measure(x, y)
    array([-0.14285714, -0.57142857,  0.10714286, -0.14285714,  0.07142857])
    >>> reliefF_measure(x, y, k_neighbors=2)
    array([-0.07142857, -0.17857143, -0.07142857, -0.0952381 , -0.17857143])
    """
    def __calc_misses(index):
        misses_diffs_classes = np.abs(
            np.vectorize(
                lambda cl: (
                        x[index]
                        - x[knn_from_class(dm, y, index, k_neighbors, cl)])
                    * prior_prob[cl],
                signature='()->(n,m)')(classes[classes != y[index]]))
        return (np.sum(np.sum(misses_diffs_classes, axis=1), axis=0)
            / (1 - prior_prob[y[index]]))

    classes, counts = np.unique(y, return_counts=True)
    if np.any(counts <= k_neighbors):
        raise ValueError(
            "Cannot calculate relieff measure because one of theclasses has "
            "less than %d samples" % (k_neighbors + 1))
    prior_prob = dict(zip(classes, np.array(counts) / len(y)))
    n_samples = x.shape[0]
    n_features = x.shape[1]
    # use manhattan distance instead of euclidean
    dm = pairwise_distances(x, x, 'manhattan')

    indices = np.arange(n_samples)
    # use abs instead of square because of manhattan distance
    hits_diffs = np.abs(
        np.vectorize(
            lambda index: (
                x[index]
                - x[knn_from_class(dm, y, index, k_neighbors, y[index])]),
            signature='()->(n,m)')(indices))
    H = np.sum(hits_diffs, axis=(0,1))

    misses_sum_diffs = np.vectorize(
        lambda index: __calc_misses(index),
        signature='()->(n)')(indices)
    M = np.sum(misses_sum_diffs, axis=0)

    weights = M - H
    # dividing by m * k guarantees that all final weights
    # will be normalized within the interval [ − 1, 1].
    weights /= n_samples * k_neighbors
    # The maximum and minimum values of A are determined over the entire
    # set of instances.
    # This normalization ensures that weight updates fall
    # between 0 and 1 for both discrete and continuous features.
    with np.errstate(divide='ignore', invalid="ignore"):  # todo
        return weights / (np.amax(x, axis=0) - np.amin(x, axis=0))


def relief_measure(x, y, m=None, random_state=42):
    """Calculate Relief measure for each feature. This measure is supposed to
    work only with binary classification datasets; for multi-class problems use
    the ReliefF measure. Bigger values mean more important features.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The input samples.
    y : array-like, shape (n_samples,)
        The classes for the samples.
    m : int, optional
        Amount of iterations to do. If not specified, n_samples iterations
        would be performed.
    random_state : int, optional
        Random state for numpy random.

    Returns
    -------
    array-like, shape (n_features,) : feature scores

    See Also
    --------
    R.J. Urbanowicz et al. Relief-based feature selection: Introduction and
    review. Journal of Biomedical Informatics 85 (2018) 189–203

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import relief_measure
    >>> import numpy as np
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1],
    ... [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 2, 2, 1, 2])
    >>> relief_measure(x, y)
    array([ 0.    , -0.6   , -0.1875, -0.15  , -0.4   ])
    """
    weights = np.zeros(x.shape[1])
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) == 1:
        raise ValueError("Cannot calculate relief measure with 1 class")
    if 1 in counts:
        raise ValueError(
            "Cannot calculate relief measure because one of the classes has "
            "only 1 sample")

    n_samples = x.shape[0]
    n_features = x.shape[1]
    if m is None:
        m = n_samples

    x_normalized = MinMaxScaler().fit_transform(x)
    dm = euclidean_distances(x_normalized, x_normalized)
    indices = np.random.default_rng(random_state).integers(
        low=0, high=n_samples, size=m)
    objects = x_normalized[indices]
    hits_diffs = np.square(
        np.vectorize(
            lambda index: (
                x_normalized[index]
                - x_normalized[knn_from_class(dm, y, index, 1, y[index])]),
            signature='()->(n,m)')(indices))
    misses_diffs = np.square(
        np.vectorize(
            lambda index: (
                x_normalized[index]
                - x_normalized[knn_from_class(
                    dm, y, index, 1, y[index], anyOtherClass=True)]),
            signature='()->(n,m)')(indices))

    H = np.sum(hits_diffs, axis=(0,1))
    M = np.sum(misses_diffs, axis=(0,1))

    weights = M - H

    return weights / m


def chi2_measure(x, y):
    """Calculate the Chi-squared measure for each feature. Bigger values mean
    more important features. This measure works best with discrete features due
    to being based on statistics.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.

    Returns
    -------
    array-like, shape (n_features,) : feature scores

    See Also
    --------
    http://lkm.fri.uni-lj.si/xaigor/slo/clanki/ijcai95z.pdf

    Example
    -------
    >>> from ITMO_FS.filters.univariate import chi2_measure
    >>> from sklearn.preprocessing import KBinsDiscretizer
    >>> import numpy as np
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1],
    ... [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
    >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
    >>> x = est.fit_transform(x)
    >>> chi2_measure(x, y)
    array([ 1.875     ,  0.83333333, 10.        ,  3.75      ,  6.66666667])
    """
    def __chi2(feature):
        values, counts = np.unique(feature, return_counts=True)
        values_map = {val: idx for idx, val in enumerate(values)}
        splits = {cl: np.array([values_map[val] for val in feature[y == cl]]) 
            for cl in classes}
        e = np.vectorize(
            lambda cl: prior_probs[cl] * counts,
            signature='()->(1)')(classes)
        n = np.vectorize(
            lambda cl: np.bincount(splits[cl], minlength=values.shape[0]),
            signature='()->(1)')(classes)
        return np.sum(np.square(e - n) / e)

    classes, counts = np.unique(y, return_counts=True)
    prior_probs = {cl: counts[idx] / x.shape[0] for idx, cl
        in enumerate(classes)}
    
    return np.apply_along_axis(__chi2, 0, x)


#
# def __contingency_matrix(labels_true, labels_pred):
#     """Build a contingency matrix describing the relationship between labels.
#         Parameters
#         ----------
#         labels_true : int array, shape = [n_samples]
#             Ground truth class labels to be used as a reference
#         labels_pred : array, shape = [n_samples]
#             Cluster labels to evaluate
#         Returns
#         -------
#         contingency : {array-like, sparse}, shape=[n_classes_true, n_classes_pred]
#             Matrix :math:`C` such that :math:`C_{i, j}` is the number of samples in
#             true class :math:`i` and in predicted class :math:`j`. If
#             ``eps is None``, the dtype of this array will be integer. If ``eps`` is
#             given, the dtype will be float.
#         """
#     classes, class_idx = np.unique(labels_true, return_inverse=True)
#     clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
#     n_classes = classes.shape[0]
#     n_clusters = clusters.shape[0]
#     # Using coo_matrix to accelerate simple histogram calculation,
#     # i.e. bins are consecutive integers
#     # Currently, coo_matrix is faster than histogram2d for simple cases
#     # TODO redo it with numpy
#     contingency = sp.coo_matrix((np.ones(class_idx.shape[0]),
#                                  (class_idx, cluster_idx)),
#                                 shape=(n_classes, n_clusters),
#                                 dtype=np.int)
#     contingency = contingency.tocsr()
#     contingency.sum_duplicates()
#     return contingency
#
#
# def __mi(U, V):
#     contingency = __contingency_matrix(U, V)
#     nzx, nzy, nz_val = sp.find(contingency)
#     contingency_sum = contingency.sum()
#     pi = np.ravel(contingency.sum(axis=1))
#     pj = np.ravel(contingency.sum(axis=0))
#     log_contingency_nm = np.log(nz_val)
#     contingency_nm = nz_val / contingency_sum
#     # Don't need to calculate the full outer product, just for non-zeroes
#     outer = (pi.take(nzx).astype(np.int64, copy=False)
#              * pj.take(nzy).astype(np.int64, copy=False))
#     log_outer = -np.log(outer) + log(pi.sum()) + log(pj.sum())
#     mi = (contingency_nm * (log_contingency_nm - log(contingency_sum)) +
#           contingency_nm * log_outer)
#     return mi.sum()
#

def spearman_corr(x, y):
    """Calculate Spearman's correlation for each feature. Bigger absolute
    values mean more important features. This measure works best with discrete
    features due to being based on statistics.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.

    Returns
    -------
    array-like, shape (n_features,) : feature scores

    See Also
    --------
    https://en.wikipedia.org/wiki/Spearman's_rank_correlation_coefficient

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import spearman_corr
    >>> from sklearn.preprocessing import KBinsDiscretizer
    >>> import numpy as np
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1],
    ... [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
    >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
    >>> x = est.fit_transform(x)
    >>> spearman_corr(x, y)
    array([-0.186339  ,  0.30429031, -0.52704628, -0.30555556,  0.35355339])
    """
    n = x.shape[0]
    if n < 2:
        raise ValueError("The input should contain more than 1 sample")

    x_ranks = np.apply_along_axis(rankdata, 0, x)
    y_ranks = rankdata(y)

    return pearson_corr(x_ranks, y_ranks)


def pearson_corr(x, y):
    """Calculate Pearson's correlation for each feature. Bigger absolute
    values mean more important features. This measure works best with discrete
    features due to being based on statistics.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.

    Returns
    -------
    array-like, shape (n_features,) : feature scores

    See Also
    --------
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import pearson_corr
    >>> from sklearn.preprocessing import KBinsDiscretizer
    >>> import numpy as np
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1],
    ... [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
    >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
    >>> x = est.fit_transform(x)
    >>> pearson_corr(x, y)
    array([-0.13363062,  0.32732684, -0.60631301, -0.26244533,  0.53452248])
    """
    x_dev = x - np.mean(x, axis=0)
    y_dev = y - np.mean(y)
    sq_dev_x = x_dev * x_dev
    sq_dev_y = y_dev * y_dev
    sum_dev = y_dev.T.dot(x_dev).reshape((x.shape[1],))
    denominators = np.sqrt(np.sum(sq_dev_y) * np.sum(sq_dev_x, axis=0))

    results = np.array(
        [(sum_dev[i] / denominators[i]) if denominators[i] > 0.0 else 0 for i
         in range(len(denominators))])
    return results


# TODO need to implement unsupervised way
def laplacian_score(x, y, k_neighbors=5, t=1, metric='euclidean', **kwargs):
    """Calculate Laplacian Score for each feature. Smaller values mean more
    important features.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The input samples.
    y : array-like, shape (n_samples,)
        The classes for the samples.
    k_neighbors : int, optional
        The number of neighbors to construct a nearest neighbor graph.
    t : float, optional
        Suitable constant for weight matrix S
        where Sij = exp(-(|xi - xj| ^ 2) / t).
    metric : str or callable, optional
        Norm function to compute distance between two points or one of the
        commonly used strings ('euclidean', 'manhattan' etc.) The default
        metric is euclidean.
    weights : array-like, shape (n_samples, n_samples)
        The weight matrix of the graph that models the local structure of
        the data space. By default it is constructed using KNN algorithm.

    Returns
    -------
    array-like, shape (n_features,) : feature scores

    See Also
    --------
    https://papers.nips.cc/paper/2909-laplacian-score-for-feature-selection.pdf

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import laplacian_score
    >>> import numpy as np
    >>> x = np.array([[1, 2, 3, 3, 1], [2, 2, 3, 3, 2], [1, 3, 3, 1, 3],
    ... [3, 1, 3, 1, 4], [4, 4, 3, 1, 5]])
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> laplacian_score(x, y)
    array([1.98983619, 1.22248371,        nan, 0.79710221, 1.90648048])
    """
    n, m = x.shape
    k_neighbors = min(k_neighbors, n - 1)
    if 'weights' in kwargs.keys():
        S = kwargs['weights']
    else:
        if n > 100000:
            S = lil_matrix((n, n))
        else:
            S = np.zeros((n, n))
        graph = NearestNeighbors(n_neighbors=k_neighbors, metric=metric)
        graph.fit(x)
        distances, neighbors = graph.kneighbors()
        for i in range(n):
            for j in range(k_neighbors):
                S[i, neighbors[i][j]] = S[neighbors[i][j], i] = exp(
                    -distances[i][j] * distances[i][j] / t)
    ONE = np.ones((n,))
    D = np.diag(S.dot(ONE))
    L = D - S
    t = D.dot(ONE)
    F = x - x.T.dot(t) / ONE.dot(t)
    F = F.T.dot(L.dot(F)) / F.T.dot(D.dot(F))
    return np.diag(F)


def information_gain(x, y):
    """Calculate mutual information for each feature by formula
    I(X,Y) = H(Y) - H(Y|X). Bigger values mean more important features. This
    measure works best with discrete features due to being based on information
    theory.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.

    Returns
    -------
    array-like, shape (n_features,) : feature scores

    See Also
    --------

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import information_gain
    >>> import numpy as np
    >>> from sklearn.preprocessing import KBinsDiscretizer
    >>> x = np.array([[1, 2, 3, 3, 1], [2, 2, 3, 3, 2], [1, 3, 3, 1, 3],
    ... [3, 1, 3, 1, 4], [4, 4, 3, 1, 5]])
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
    >>> x = est.fit_transform(x)
    >>> information_gain(x, y)
    array([1.33217904, 1.33217904, 0.        , 0.67301167, 1.60943791])
    """
    entropy_x = entropy(y)
    cond_entropy = np.apply_along_axis(conditional_entropy, 0, x, y)
    return entropy_x - cond_entropy


def anova(x, y):
    """Calculate anova measure for each feature. Bigger values mean more
    important features.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.

    Returns
    -------
    array-like, shape (n_features,) : feature scores

    See Also
    --------
    Lowry, Richard.  "Concepts and Applications of Inferential Statistics".
    Chapter 14. http://vassarstats.net/textbook/

    Note:
    The Anova score is counted for checking hypothesis if variances of two
    samples are similar, this measure only returns you counted F-score.
    For understanding whether samples' variances are similar you should
    compare recieved result with value of F-distribution function, for
    example use:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.fdtrc.html#scipy.special.fdtrc

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import anova
    >>> import numpy as np
    >>> x = np.array([[1, 2, 3, 3, 1], [2, 2, 3, 3, 2], [1, 3, 3, 1, 3],
    ... [3, 1, 3, 1, 4], [4, 4, 3, 1, 5]])
    >>> y = np.array([1, 2, 1, 3, 3])
    >>> anova(x, y)
    array([12.6 ,  0.04,   nan,  1.4 ,  3.  ])
    """
    split_by_class = [x[y == k] for k in np.unique(y)]
    num_classes = len(np.unique(y))
    num_samples = x.shape[0]
    num_samples_by_class = [s.shape[0] for s in split_by_class]
    sq_sum_all = sum((s ** 2).sum(axis=0) for s in split_by_class)
    sum_group = [np.asarray(s.sum(axis=0)) for s in split_by_class]
    sq_sum_combined = sum(sum_group) ** 2
    sum_sq_group = [np.asarray((s ** 2).sum(axis=0)) for s in split_by_class]
    sq_sum_group = [s ** 2 for s in sum_group]
    sq_sum_total = sq_sum_all - sq_sum_combined / float(num_samples)
    sq_sum_within = sum(
        [sum_sq_group[i] - sq_sum_group[i] / num_samples_by_class[i] for i in
         range(num_classes)])
    sq_sum_between = sq_sum_total - sq_sum_within
    deg_free_between = num_classes - 1
    deg_free_within = num_samples - num_classes
    ms_between = sq_sum_between / float(deg_free_between)
    ms_within = sq_sum_within / float(deg_free_within)
    f = ms_between / ms_within
    return np.array(f)


def modified_t_score(x, y):
    """Calculate the Modified T-score for each feature. Bigger values mean
    more important features.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The input samples.
    y : array-like, shape (n_samples,)
        The classes for the samples. There can be only 2 classes.

    Returns
    -------
    array-like, shape (n_features,) : feature scores

    See Also
    --------
    For more details see paper <https://dergipark.org.tr/en/download/article-file/261247>.

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import modified_t_score
    >>> import numpy as np
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1],
    ... [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 1, 2, 1, 2])
    >>> modified_t_score(x, y)
    array([1.68968099, 0.12148022, 0.39653932, 0.17682997, 2.04387142])
    """
    classes = np.unique(y)

    size_class0 = y[y == classes[0]].size
    size_class1 = y[y == classes[1]].size

    mean_class0 = np.mean(x[y == classes[0]], axis=0)
    mean_class0 = np.nan_to_num(mean_class0)
    mean_class1 = np.mean(x[y == classes[1]], axis=0)
    mean_class1 = np.nan_to_num(mean_class1)

    std_class0 = np.std(x[y == classes[0]], axis=0)
    std_class0 = np.nan_to_num(std_class0)
    std_class1 = np.std(x[y == classes[1]], axis=0)
    std_class1 = np.nan_to_num(std_class1)

    corr_with_y = np.apply_along_axis(
        lambda feature: abs(np.corrcoef(feature, y)[0][1]), 0, x)
    corr_with_y = np.nan_to_num(corr_with_y)

    corr_with_others = abs(np.corrcoef(x, rowvar=False))
    corr_with_others = np.nan_to_num(corr_with_others)

    mean_of_corr_with_others = (
        corr_with_others.sum(axis=1)
        - corr_with_others.diagonal()) / (len(corr_with_others) - 1)

    t_score_numerator = abs(mean_class0 - mean_class1)
    t_score_denominator = np.sqrt(
        (size_class0 * np.square(std_class0) + size_class1 * np.square(
            std_class1)) / (size_class0 + size_class1))
    modificator = corr_with_y / mean_of_corr_with_others

    modified_t_score = t_score_numerator / t_score_denominator * modificator
    modified_t_score = np.nan_to_num(modified_t_score)

    return modified_t_score


MEASURE_NAMES = {"FitCriterion": fit_criterion_measure,
                 "FRatio": f_ratio_measure,
                 "GiniIndex": gini_index,
                 "SymmetricUncertainty": su_measure,
                 "SpearmanCorr": spearman_corr,
                 "PearsonCorr": pearson_corr,
                 "FechnerCorr": fechner_corr,
                 "KendallCorr": kendall_corr,
                 "ReliefF": reliefF_measure,
                 "Chi2": chi2_measure,
                 "Anova": anova,
                 "LaplacianScore": laplacian_score,
                 "InformationGain": information_gain,
                 "ModifiedTScore": modified_t_score,
                 "Relief": relief_measure}


def select_best_by_value(value):
    return _wrapped_partial(__select_by_value, value=value, more=True)


def select_worst_by_value(value):
    return _wrapped_partial(__select_by_value, value=value, more=False)


def __select_by_value(scores, value, more=True):
    if more:
        return np.flatnonzero(scores >= value)
    else:
        return np.flatnonzero(scores <= value)


def select_k_best(k):
    return _wrapped_partial(__select_k, k=k, reverse=True)


def select_k_worst(k):
    return _wrapped_partial(__select_k, k=k)


def __select_k(scores, k, reverse=False):
    if not isinstance(k, int):
        raise TypeError("Number of features should be integer")
    if k > scores.shape[0]:
        raise ValueError(
            "Cannot select %d features with n_features = %d" % (k, len(scores)))
    order = np.argsort(scores)
    if reverse:
        order = order[::-1]
    return order[:k]


def __select_percentage_best(scores, percent):
    return __select_k(
        scores, k=(int)(scores.shape[0] * percent), reverse=True)


def select_best_percentage(percent):
    return _wrapped_partial(__select_percentage_best, percent=percent)


def __select_percentage_worst(scores, percent):
    return __select_k(
        scores, k=(int)(scores.shape[0] * percent), reverse=False)


def select_worst_percentage(percent):
    return _wrapped_partial(__select_percentage_worst, percent=percent)


CR_NAMES = {"Best by value": select_best_by_value,
            "Worst by value": select_worst_by_value,
            "K best": select_k_best,
            "K worst": select_k_worst,
            "Worst by percentage": select_worst_percentage,
            "Best by percentage": select_best_percentage}


def qpfs_filter(X, y, r=None, sigma=None, solv='quadprog', fn=pearson_corr):
    """Performs Quadratic Programming Feature Selection algorithm.
    Note: this realization requires labels to start from 1 and be numerical.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input samples.
    y : array-like, shape (n_samples,)
        The classes for the samples.
    r : int
        The number of samples to be used in Nystrom optimization.
    sigma : double
        The threshold for eigenvalues to be used in solving QP optimization.
    solv : string, default
        The name of qp solver according to
        qpsolvers(https://pypi.org/project/qpsolvers/) naming. Note quadprog
        is used by default.
    fn : function(array, array), default
        The function to count correlation, for example pierson correlation or
        mutual information. Note mutual information is used by default.

    Returns
    -------
    array-like, shape (n_features,) : the ranks of features in dataset, with
    rank increase, feature relevance increases and redundancy decreases.

    See Also
    --------
    http://www.jmlr.org/papers/volume11/rodriguez-lujan10a/rodriguez-lujan10a.pdf

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import qpfs_filter
    >>> from sklearn.datasets import make_classification
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1],
    ... [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
    >>> ranks = qpfs_filter(x, y)
    >>> print(ranks)
    """
    return qpfs_body(X, y, fn, r=r, sigma=sigma, solv=solv)


from logging import getLogger

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

# from ...utils import l21_norm, matrix_norm, power_neg_half, BaseTransformer


class NDFS(BaseTransformer):
    """Nonnegative Discriminative Feature Selection algorithm.

    Parameters
    ----------
    n_features : int
        Number of features to select.
    c : int
        Amount of clusters to find.
    k : int
        Amount of nearest neighbors to use while building the graph.
    alpha : float
        Parameter in the objective function.
    beta : float
        Regularization parameter in the objective function.
    gamma : float
        Parameter in the objective function that controls the orthogonality
        condition.
    sigma : float
        Parameter for the weighting scheme.
    max_iterations : int
        Maximum amount of iterations to perform.
    epsilon : positive float
        Specifies the needed residual between the target functions from
        consecutive iterations. If the residual is smaller than epsilon, the
        algorithm is considered to have converged.

    See Also
    --------
    http://www.nlpr.ia.ac.cn/2012papers/gjhy/gh27.pdf

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import NDFS
    >>> import numpy as np
    >>> X = np.array([[1, 2, 3, 3, 1], [2, 2, 3, 3, 2], [1, 3, 3, 1, 3],
    ... [1, 1, 3, 1, 4], [2, 4, 3, 1, 5]])
    >>> y = np.array([1, 2, 1, 1, 2])
    >>> model = NDFS(3).fit(X, y)
    >>> model.selected_features_
    array([0, 3, 4], dtype=int64)
    >>> model = NDFS(3).fit(X)
    >>> model.selected_features_
    array([3, 4, 1], dtype=int64)
    """
    def __init__(self, n_features, c=2, k=3, alpha=1, beta=1, gamma=10e8,
                 sigma=1, max_iterations=1000, epsilon=1e-5):
        self.n_features = n_features
        self.c = c
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.max_iterations = max_iterations
        self.epsilon = epsilon

    def __scheme(self, x1, x2):
        return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (self.sigma ** 2))

    def _fit(self, X, y, **kwargs):
        """Fit the filter.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_classes)
            The target values or their one-hot encoding that are used to
            compute F. If not present, a k-means clusterization algorithm
            is used. If present, n_classes should be equal to c.

        Returns
        -------
        None
        """
        n_samples = X.shape[0]

        if self.k >= n_samples:
            getLogger(__name__).error(
                "Cannot select %d nearest neighbors with n_samples = %d",
                self.k, n_samples)
            raise ValueError(
                "Cannot select %d nearest neighbors with n_samples = %d"
                % (self.k, n_samples))

        graph = NearestNeighbors(
            n_neighbors=self.k,
            algorithm='ball_tree').fit(X).kneighbors_graph().toarray()
        graph = np.minimum(1, graph + graph.T)
        getLogger(__name__).info("Nearest neighbors graph: %s", graph)

        S = graph * pairwise_distances(
            X, metric=lambda x, y: self.__scheme(x, y))
        getLogger(__name__).info("S: %s", S)
        A = np.diag(S.sum(axis=0))
        getLogger(__name__).info("A: %s", A)
        L = power_neg_half(A).dot(A - S).dot(power_neg_half(A))
        getLogger(__name__).info("L: %s", L)

        if y is not None:
            if len(y.shape) == 2:
                Y = y
            else:
                Y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
        else:
            if self.c > n_samples:
                getLogger(__name__).error(
                    "Cannot find %d clusters with n_samples = %d", self.c,
                    n_samples)
                raise ValueError(
                    "Cannot find %d clusters with n_samples = %d"
                    % (self.c, n_samples))
            Y = self.__run_kmeans(X)
        getLogger(__name__).info("Transformed Y: %s", Y)
        F = Y.dot(power_neg_half(Y.T.dot(Y)))
        getLogger(__name__).info("F: %s", F)
        D = np.eye(self.n_features_)
        In = np.eye(n_samples)
        Ic = np.eye(Y.shape[1])

        previous_target = -1
        for _ in range(self.max_iterations):
            M = (L + self.alpha
                * (In - X.dot(
                    np.linalg.inv(X.T.dot(X) + self.beta * D)).dot(X.T)))
            getLogger(__name__).info("M: %s", M)
            F = (F * ((self.gamma * F)
                       / (M.dot(F) + self.gamma * F.dot(F.T).dot(F))))
            getLogger(__name__).info("F: %s", F)
            W = np.linalg.inv(X.T.dot(X) + self.beta * D).dot(X.T.dot(F))
            getLogger(__name__).info("W: %s", W)
            diag = 2 * matrix_norm(W)
            diag[diag < 1e-10] = 1e-10  # prevents division by zero
            D = np.diag(1 / diag)
            getLogger(__name__).info("D: %s", D)

            target = (np.trace(F.T.dot(L).dot(F))
                + self.alpha * (np.linalg.norm(X.dot(W) - F) ** 2
                    + self.beta * l21_norm(W))
                + self.gamma * (np.linalg.norm(F.T.dot(F) - Ic) ** 2) / 2)
            getLogger(__name__).info("New target value: %d", target)
            if abs(target - previous_target) < self.epsilon:
                break
            previous_target = target

        getLogger(__name__).info("Ended up with W: %s", W)
        self.feature_scores_ = matrix_norm(W)
        getLogger(__name__).info("Feature scores: %s", self.feature_scores_)
        ranking = np.argsort(self.feature_scores_)[::-1]
        self.selected_features_ = ranking[:self.n_features]

    def __run_kmeans(self, X):
        kmeans = KMeans(n_clusters=self.c, copy_x=True)
        kmeans.fit(X)
        labels = kmeans.labels_
        getLogger(__name__).info("Labels from KMeans: %s", labels)
        return OneHotEncoder().fit_transform(labels.reshape(-1, 1)).toarray()



from logging import getLogger

import numpy as np
from scipy.linalg import eigh
from sklearn.metrics.pairwise import pairwise_distances

# from ...utils import l21_norm, matrix_norm, power_neg_half, BaseTransformer


class SPEC(BaseTransformer):
    """Spectral Feature Selection algorithm.

    Parameters
    ----------
    n_features : int
        Number of features to select.
    k : int
        Amount of clusters to find.
    gamma : callable
        An "increasing function that penalizes high frequency components".
        Default is gamma(x) = x^2.
    sigma : float
        Parameter for the weighting scheme.
    phi_type : int (1, 2 or 3)
        Type of feature ranking function to use.

    Notes
    -----
    For more details see `this paper <http://www.public.asu.edu/~huanliu/papers/icml07.pdf/>`_.

    Examples
    --------
    >>> from ITMO_FS.filters.univariate import SPEC
    >>> import numpy as np
    >>> X = np.array([[1, 2, 3, 3, 1], [2, 2, 3, 3, 2], [1, 3, 3, 1, 3],
    ... [1, 1, 3, 1, 4], [2, 4, 3, 1, 5]])
    >>> y = np.array([1, 2, 1, 1, 2])
    >>> model = SPEC(3).fit(X, y)
    >>> model.selected_features_
    array([0, 1, 4], dtype=int64)
    >>> model = SPEC(3).fit(X)
    >>> model.selected_features_
    array([3, 4, 1], dtype=int64)
    """
    def __init__(self, n_features, k=2, gamma=(lambda x: x ** 2), sigma=0.5,
                 phi_type=3):
        self.n_features = n_features
        self.k = k
        self.gamma = gamma
        self.sigma = sigma
        self.phi_type = phi_type

    def __scheme(self, x1, x2):
        return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * self.sigma ** 2))

    def __phi1(self, cosines, eigvals, k):
        return np.sum(cosines * cosines * self.gamma(eigvals))

    def __phi2(self, cosines, eigvals, k):
        return (np.sum(cosines[1:] * cosines[1:] * self.gamma(eigvals[1:]))
                / np.sum(cosines[1:] * cosines[1:]))

    def __phi3(self, cosines, eigvals, k):
        return np.sum(cosines[1:k] * cosines[1:k]
                      * (self.gamma(2) - self.gamma(eigvals[1:k])))

    def _fit(self, X, y):
        """Fit the filter.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,), optional
            The target values. If present, label values are used to
            construct the similarity graph and the amount of classes
            overrides k.

        Returns
        -------
        None
        """
        def calc_weight(f):
            f_norm = np.sqrt(D).dot(f)
            f_norm /= np.linalg.norm(f_norm)

            cosines = np.apply_along_axis(
                lambda vec: np.dot(vec / np.linalg.norm(vec), f_norm), 0,
                eigvectors)
            return phi(cosines, eigvals, k)

        if self.phi_type == 1:
            phi = self.__phi1
        elif self.phi_type == 2:
            phi = self.__phi2
        elif self.phi_type == 3:
            phi = self.__phi3
        else:
            getLogger(__name__).error(
                "phi_type should be 1, 2 or 3, %d passed", self.phi_type)
            raise ValueError(
                "phi_type should be 1, 2 or 3, %d passed" % self.phi_type)

        n_samples = X.shape[0]

        if y is None:
            if self.k > n_samples:
                getLogger(__name__).error(
                    "Cannot find %d clusters with n_samples = %d",
                    self.k, n_samples)
                raise ValueError(
                    "Cannot find %d clusters with n_samples = %d"
                    % (self.k, n_samples))
            k = self.k
            graph = np.ones((n_samples, n_samples))
            W = graph * pairwise_distances(
                X, metric=lambda x, y: self.__scheme(x, y))
        else:
            values, counts = np.unique(y, return_counts=True)
            values_dict = dict(zip(values, counts))
            k = len(values)
            W = pairwise_distances(
                y.reshape(-1, 1),
                metric=lambda x, y: (x[0] == y[0]) / values_dict[x[0]])

        getLogger(__name__).info("W: %s", W)

        D = np.diag(W.sum(axis=1))
        getLogger(__name__).info("D: %s", D)
        L = D - W
        getLogger(__name__).info("L: %s", L)
        L_norm = power_neg_half(D).dot(L).dot(power_neg_half(D))
        getLogger(__name__).info("Normalized L: %s", L_norm)
        eigvals, eigvectors = eigh(a=L_norm)
        getLogger(__name__).info(
            "Eigenvalues for normalized L: %s, eigenvectors: %s",
            eigvals, eigvectors)

        self.feature_scores_ = np.apply_along_axis(
            lambda f: calc_weight(f), 0, X)
        getLogger(__name__).info("Feature scores: %s", self.feature_scores_)
        ranking = np.argsort(self.feature_scores_)
        if self.phi_type == 3:
            ranking = ranking[::-1]
        self.selected_features_ = ranking[:self.n_features]



from logging import getLogger

import numpy as np

# from .measures import CR_NAMES, MEASURE_NAMES
# from ...utils import (BaseTransformer, generate_features, check_restrictions,
                    #   apply_cr)


class UnivariateFilter(BaseTransformer):
    """Basic interface for using univariate measures for feature selection.
    List of available measures is in ITMO_FS.filters.univariate.measures, also
    you can provide your own measure but it should suit the argument scheme for
    measures, i.e. take two arguments x,y and return scores for all the
    features in dataset x. Same applies to cutting rules.

    Parameters
    ----------
    measure : string or callable
        A metric name defined in GLOB_MEASURE or a callable with signature
        measure (sample dataset, labels of dataset samples) which should
        return a list of metric values for each feature in the dataset.
    cutting_rule : string or callables
        A cutting rule name defined in GLOB_CR or a callable with signature
        cutting_rule (features) which should return a list of features ranked by
        some rule.

    See Also
    --------

    Examples
    --------

    >>> import numpy as np
    >>> from ITMO_FS.filters.univariate import select_k_best
    >>> from ITMO_FS.filters.univariate import UnivariateFilter
    >>> from ITMO_FS.filters.univariate import f_ratio_measure
    >>> x = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1],
    ... [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 3, 2, 1, 2])
    >>> filter = UnivariateFilter(f_ratio_measure,
    ... select_k_best(2)).fit(x, y)
    >>> filter.selected_features_
    array([4, 2], dtype=int64)
    >>> filter.feature_scores_
    array([0.6 , 0.2 , 1.  , 0.12, 5.4 ])
    """
    def __init__(self, measure, cutting_rule=("Best by percentage", 1.0)):
        self.measure = measure
        self.cutting_rule = cutting_rule

    def __apply_ms(self):
        if isinstance(self.measure, str):
            try:
                measure = MEASURE_NAMES[self.measure]
            except KeyError:
                getLogger(__name__).error("No %s measure yet", self.measure)
                raise KeyError("No %s measure yet" % self.measure)
        elif hasattr(self.measure, '__call__'):
            measure = self.measure
        else:
            getLogger(__name__).error(
                "%s isn't a measure function or string", self.measure)
            raise KeyError(
                "%s isn't a measure function or string" % self.measure)
        return measure

    def _fit(self, X, y, store_scores=True):
        """Fit the filter.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values.
        store_scores : boolean, optional
            In case you want to store the scores of features
            for future calls to Univariate filter; default True

        Returns
        -------
        None
        """
        measure = self.__apply_ms()
        cutting_rule = apply_cr(self.cutting_rule)
        getLogger(__name__).info(
            "Using UnivariateFilter with measure %s and cutting rule %s",
            measure, cutting_rule)

        check_restrictions(measure.__name__, cutting_rule.__name__)

        feature_scores = measure(X, y)
        getLogger(__name__).info("Feature scores: %s", feature_scores)

        if store_scores:
            self.feature_scores_ = feature_scores
        self.selected_features_ = cutting_rule(feature_scores)


import numpy as np

from ITMO_FS.utils.functions import cartesian
# from ...utils import BaseTransformer


#  TODO some optimization and sklearn-like API
class VDM(BaseTransformer):
    """
        Creates Value Difference Metric builder.
        For continious features discretesation requered.

        Parameters
        ----------
        weighted: bool
            If weighted = False, modified version of metric which omits the
            weights is used
        q: int
            Power in VDM usually 1 or 2

        Notes
        -----
        For more details see papers about
        `Improved Heterogeneous Distance Functions
        <https://www.jair.org/index.php/jair/article/view/10182/>`_
        and `Implicit Future Selection with the VDM
        <https://aura.abdn.ac.uk/bitstream/handle/2164/10951/payne_ecai_98.pdf?sequence=1/>`_.

        Examples
        --------
        >>> x = np.array([[0, 0, 0, 0],
        ...               [1, 0, 1, 1],
        ...               [1, 0, 0, 2]])
        >>> y = np.array([0,
        ...               1,
        ...               1])
        >>> vdm = VDM()
        >>> vdm.fit(x, y)
        array([[0.         4.35355339 4.        ]
               [4.5        0.         0.5       ]
               [4.         0.35355339 0.        ]])
    """

    def __init__(self, weighted=True, q=1):
        self.weighted = weighted
        self.q = q

    def _fit(self, X, y=None, **kwargs):
        """
            Generates metric for the data
            Complexity: O(n_features * n_samples^3) worst case, should be
            faster on a real data.

            Parameters
            ----------
            X: array-like, shape (n_features, n_samples)
                Input samples' parameters. Parameters among every class must be
                sequential integers.
            y: array-like, shape (n_samples)
                Input samples' class labels. Class labels must be sequential
                integers.
            Returns
            -------
            result:
                numpy.ndarray, shape=(n_samples, n_samples), dtype=np.double
                with selected version of metrics
            See Also
            --------
        """
        # TODO Fix case of y passed as DataFrame. For now y is transformed
        #  to 2D array and this causes an error. It seems better to follow
        #  usual sklearn practice and to use check_X_y but np.asarray(y[0])
        #  is also possible
        n_labels = np.max(y) + 1  # Number of different class labels
        n_samples = X.shape[0]  # Number of samples

        vdm = np.zeros((n_samples, n_samples),
                       dtype=np.double)  # Initializing output matrix

        for feature in X.T:  # For each attribute:
            # Initializing utility structures:
            n_values = np.max(
                feature) + 1  # Number of different values for the feature

            entries_x = np.empty(n_values,
                                 dtype=object)  # Array containing list of
            # indexes for every feature value
            entries_x[:] = [[] for _ in range(n_values)]

            entries_c_x = np.array(
                [{} for _ in range(n_labels)])  # Array of dirs of kind
            # {(feature value, amount of entries) for each class label

            for i, value in enumerate(feature):  # For each sample:
                entries_x[value].append(
                    i)  # Adding sample index to entries list
                entries_c_x[y[i]][value] = entries_c_x[y[i]].get(value,
                                                                 0) + 1  #
                # Adding entry for corresponding
                # class label

            amounts_x = np.array(list(
                map(len, entries_x)))  # Array containing amounts of samples
            # for every feature value

            # Calculating deltas:

            deltas = np.zeros((n_values, n_values),
                              dtype=np.double)  # Array for calculating deltas

            # Calculating components where exactly one of probabilities is
            # not zero:
            for c in range(n_labels):  # For each class:
                entries = np.array(list(entries_c_x[
                                            c].keys()))  # Feature values
                # which are presented in pairs for
                # the class
                amounts = np.array(
                    list(entries_c_x[c].values()))  # Corresponding amounts
                non_entries = np.arange(
                    n_values)  # Feature values which are not presented in pairs for the class
                # TODO get rid of error if entries are empty, example in test
                non_entries[entries] = -1
                non_entries = non_entries[non_entries != -1]

                for i in range(len(entries)):  # For each feature value
                    value = entries[i]  # Current value
                    v_c_instances = amounts[
                        i]  # Amount of instances with such value and such class
                    v_instances = amounts_x[
                        value]  # Amount of instances with such value
                    target_x, target_y = cartesian([value],
                                                   non_entries)  # Target indexes for deltas array
                    deltas[target_x, target_y] += (
                                                          v_c_instances / v_instances) ** 2
            deltas += deltas.T  # As we didn't determined indexes order, for each i, j some components are
            # written to delta(i, j) while others to delta(j, i), but exactly once. Adding transposed matrix to fix this

            # Calculating components where both probabilities are not zero:
            for c in range(n_labels):  # For each class:
                entries = np.array(list(entries_c_x[
                                            c].keys()))  # Feature values which are presented in pairs for
                # the class
                amounts = np.array(
                    list(entries_c_x[c].values()))  # Corresponding amounts
                probs = amounts / amounts_x[
                    entries]  # Conditional probabilities
                target_x, target_y = cartesian(np.arange(len(entries)),
                                               np.arange(len(
                                                   entries)))  # Target indexes
                # for deltas array
                deltas[entries[target_x], entries[target_y]] += (probs[
                                                                     target_x] -
                                                                 probs[
                                                                     target_y]) ** 2

            # Updating vdm:
            if not self.weighted:  # If non-weighted version of metrics was selected
                for i in range(n_values):  # For each value i
                    for j in range(n_values):  # For each value j
                        if amounts_x[i] == 0 or amounts_x[
                            j] == 0:  # If some value does not appear in current feature,
                            # skip it
                            continue
                        vdm[cartesian(entries_x[i], entries_x[j])] += \
                            deltas[i][j]
            else:  # If weighted version of metrics was selected
                weights = np.zeros(n_values,
                                   dtype=np.double)  # Initializing weights array
                for c in range(n_labels):  # For each class:
                    entries = np.array(list(entries_c_x[
                                                c].keys()))  # Feature values which are presented in pairs for
                    # the class
                    amounts = np.array(
                        list(entries_c_x[c].values()))  # Corresponding amounts
                    probs = amounts / amounts_x[
                        entries]  # Conditional probabilities
                    weights[entries] += probs ** 2
                weights = np.sqrt(weights)

                for i in range(n_values):  # For each value i
                    for j in range(n_values):  # For each value j
                        if amounts_x[i] == 0 or amounts_x[j] == 0:
                            continue
                        vdm[cartesian(entries_x[i], entries_x[j])] += \
                            deltas[i][j] * weights[i]

        return vdm


###########################################################################
from logging import getLogger

import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import Lars
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances

# from ...utils import BaseTransformer


class MCFS(BaseTransformer):
    """Unsupervised Feature Selection for Multi-Cluster Data algorithm.

    Parameters
    ----------
    n_features : int
        Number of features to select.
    k : int
        Amount of clusters to find.
    p : int
        Amount of nearest neighbors to use while building the graph.
    scheme : str, either '0-1', 'heat' or 'dot'
        Weighting scheme to use while building the graph.
    sigma : float
        Parameter for heat weighting scheme. Ignored if scheme is not 'heat'.
    full_graph : boolean
        If True, connect all vertices in the graph to each other instead of
        running the k-nearest neighbors algorithm. Use with 'heat' or 'dot'
        schemes.

    Notes
    -----
    For more details see `this paper
    <http://www.cad.zju.edu.cn/home/dengcai/Publication/Conference/2010_KDD-MCFS.pdf/>`_.

    Examples
    --------
    >>> from ITMO_FS.filters.unsupervised import MCFS
    >>> from sklearn.datasets import make_classification
    >>> import numpy as np
    >>> dataset = make_classification(n_samples=500, n_features=100,
    ... n_informative=5, n_redundant=0, random_state=42, shuffle=False)
    >>> X, y = np.array(dataset[0]), np.array(dataset[1])
    >>> model = MCFS(5).fit(X)
    >>> model.selected_features_
    array([0, 2, 4, 1, 3], dtype=int64)
    """
    def __init__(self, n_features, k=2, p=3, scheme='dot', sigma=1,
                 full_graph=False):
        self.n_features = n_features
        self.k = k
        self.p = p
        self.scheme = scheme
        self.sigma = sigma
        self.full_graph = full_graph

    def __scheme_01(self, x1, x2):
        return 1

    def __scheme_heat(self, x1, x2):
        return np.exp(-np.linalg.norm(x1 - x2) ** 2 / self.sigma)

    def __scheme_dot(self, x1, x2):
        return (x1 / np.linalg.norm(x1 + 1e-10)).dot(
            x2 / np.linalg.norm(x2 + 1e-10))

    def _fit(self, X, y):
        """
            Fits the filter.

            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                The training input samples.
            y : array-like
                The target values (ignored).

            Returns
            ----------
            None
        """
        if self.scheme == '0-1':
            scheme = self.__scheme_01
        elif self.scheme == 'heat':
            scheme = self.__scheme_heat
        elif self.scheme == 'dot':
            scheme = self.__scheme_dot
        else:
            getLogger(__name__).error(
                "scheme should be either '0-1', 'heat' or 'dot'; %s passed",
                self.scheme)
            raise KeyError(
                "scheme should be either '0-1', 'heat' or 'dot'; %s passed"
                % self.scheme)

        n_samples = X.shape[0]


        if self.k > n_samples:
            getLogger(__name__).error(
                "Cannot find %d clusters with n_samples = %d",
                self.k, n_samples)
            raise ValueError(
                "Cannot find %d clusters with n_samples = %d"
                % (self.k, n_samples))

        if self.p >= n_samples:
            getLogger(__name__).error(
                "Cannot select %d nearest neighbors with n_samples = %d",
                self.p, n_samples)
            raise ValueError(
                "Cannot select %d nearest neighbors with n_samples = %d"
                % (self.p, n_samples))

        if self.full_graph:
            graph = np.ones((n_samples, n_samples))
        else:
            graph = NearestNeighbors(n_neighbors=self.p,
                algorithm='ball_tree').fit(X).kneighbors_graph().toarray()
            graph = np.minimum(1, graph + graph.T)

        getLogger(__name__).info("Nearest neighbors graph: %s", graph)

        W = graph * pairwise_distances(X, metric=lambda x, y: scheme(x, y))
        getLogger(__name__).info("W: %s", W)
        D = np.diag(W.sum(axis=0))
        getLogger(__name__).info("D: %s", D)
        L = D - W
        getLogger(__name__).info("L: %s", L)
        eigvals, Y = eigh(type=1, a=L, b=D, subset_by_index=[1, self.k])
        getLogger(__name__).info("Eigenvalues: %s, classes: %s", eigvals, Y)

        weights = np.zeros((self.n_features_, self.k))
        for i in range(self.k):
            clf = Lars(n_nonzero_coefs=self.n_features)
            clf.fit(X, Y[:, i])
            weights[:, i] = np.abs(clf.coef_)
            getLogger(__name__).info(
                "Weights for eigenvalue %d: %s", i, weights[:, i])

        self.feature_scores_ = weights.max(axis=1)
        getLogger(__name__).info("Feature scores: %s", self.feature_scores_)
        ranking = np.argsort(self.feature_scores_)[::-1]
        self.selected_features_ = ranking[:self.n_features]


from logging import getLogger

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors

# from ...utils import BaseTransformer

class TraceRatioLaplacian(BaseTransformer):
    """TraceRatio(similarity based) feature selection filter performed in
    unsupervised way, i.e laplacian version

    Parameters
    ----------
    n_features : int
        Amount of features to select.
    k : int
        Amount of nearest neighbors to use while building the graph.
    t : int
        constant for kernel function calculation
    epsilon : float
        Lambda change threshold.

    Notes
    -----
    For more details see `this paper <https://aaai.org/Papers/AAAI/2008/AAAI08-107.pdf/>`_.

    Examples
    --------
    >>> from ITMO_FS.filters.unsupervised import TraceRatioLaplacian
    >>> import numpy as np
    >>> X = np.array([[1, 2, 3, 3, 1], [2, 2, 3, 3, 2], [1, 3, 3, 1, 3],
    ... [1, 1, 3, 1, 4], [2, 4, 3, 1, 5]])
    >>> y = np.array([1, 2, 1, 1, 2])
    >>> tracer = TraceRatioLaplacian(2, k=2).fit(X)
    >>> tracer.selected_features_
    array([3, 1], dtype=int64)
    """
    def __init__(self, n_features, k=5, t=1, epsilon=1e-3):
        self.n_features = n_features
        self.k = k
        self.t = t
        self.epsilon = epsilon

    def _fit(self, X, y):
        """Fit the filter.

        Parameters
        ----------
        X : array-likey, shape (n_samples, n_features)
            The training input samples
        y : array-like, shape (n_samples,)
            The target values

        Returns
        -------
        None
        """
        n_samples = X.shape[0]

        if self.k >= n_samples:
            getLogger(__name__).error(
                "Cannot select %d nearest neighbors with n_samples = %d",
                self.k, n_samples)
            raise ValueError(
                "Cannot select %d nearest neighbors with n_samples = %d"
                % (self.k, n_samples))

        graph = NearestNeighbors(
            n_neighbors=self.n_features,
            algorithm='ball_tree').fit(X).kneighbors_graph().toarray()
        graph = np.minimum(1, graph + graph.T)
        getLogger(__name__).info("Nearest neighbors graph: %s", graph)

        A_within = graph * pairwise_distances(
            X, metric=lambda x, y: np.exp(-np.linalg.norm(x - y) ** 2 / self.t))
        getLogger(__name__).info("A_within: %s", A_within)
        D_within = np.diag(A_within.sum(axis=1))
        getLogger(__name__).info("D_within: %s", D_within)
        L_within = D_within - A_within
        getLogger(__name__).info("L_within: %s", L_within)
        A_between = (D_within.dot(np.ones((n_samples, n_samples))).dot(D_within)
                     / np.sum(D_within))
        getLogger(__name__).info("A_between: %s", A_between)
        D_between = np.diag(A_between.sum(axis=1))
        getLogger(__name__).info("D_between: %s", D_between)
        L_between = D_between - A_between
        getLogger(__name__).info("L_between: %s", L_between)

        E = X.T.dot(L_within).dot(X)
        B = X.T.dot(L_between).dot(X)

        # we need only diagonal elements for trace calculation
        e = np.array(np.diag(E))
        b = np.array(np.diag(B))
        getLogger(__name__).info("E: %s", e)
        getLogger(__name__).info("B: %s", b)
        lam = 0
        prev_lam = -1
        while lam - prev_lam >= self.epsilon:  # TODO: optimize
            score = b - lam * e
            getLogger(__name__).info("Score: %s", score)
            self.selected_features_ = np.argsort(score)[::-1][:self.n_features]
            getLogger(__name__).info(
                "New selected set: %s", self.selected_features_)
            prev_lam = lam
            lam = (np.sum(b[self.selected_features_])
                   / np.sum(e[self.selected_features_]))
            getLogger(__name__).info("New lambda: %d", lam)
        self.score_ = score
        self.lam_ = lam


from logging import getLogger

import numpy as np
from scipy.linalg import eigh
from sklearn.neighbors import NearestNeighbors

# from ...utils import l21_norm, matrix_norm, BaseTransformer


class UDFS(BaseTransformer):
    """Unsupervised Discriminative Feature Selection algorithm.

    Parameters
    ----------
    n_features : int
        Number of features to select.
    c : int
        Amount of clusters to find.
    k : int
        Amount of nearest neighbors to use while building the graph.
    gamma : float
        Regularization term in the target function.
    l : float
        Parameter that controls the invertibility of the matrix used in
        computing of B.
    max_iterations : int
        Maximum amount of iterations to perform.
    epsilon : positive float
        Specifies the needed residual between the target functions from
        consecutive iterations. If the residual is smaller than epsilon, the
        algorithm is considered to have converged.

    Notes
    -----
    For more details see `this paper <https://www.ijcai.org/Proceedings/11/Papers/267.pdf/>`_.

    Examples
    --------
    >>> from ITMO_FS.filters.unsupervised import UDFS
    >>> from sklearn.datasets import make_classification
    >>> import numpy as np
    >>> dataset = make_classification(n_samples=500, n_features=100,
    ... n_informative=5, n_redundant=0, random_state=42, shuffle=False,
    ... n_clusters_per_class=1)
    >>> X, y = np.array(dataset[0]), np.array(dataset[1])
    >>> model = UDFS(5).fit(X)
    >>> model.selected_features_
    array([ 2,  3, 19, 90, 92], dtype=int64)
    """
    def __init__(self, n_features, c=2, k=3, gamma=1, l=1e-6,
                 max_iterations=1000, epsilon=1e-5):
        self.n_features = n_features
        self.c = c
        self.k = k
        self.gamma = gamma
        self.l = l
        self.max_iterations = max_iterations
        self.epsilon = epsilon

    def _fit(self, X, y):
        """Fit the filter.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like
            The target values (ignored).

        Returns
        -------
        None
        """
        def construct_S(arr):
            S = np.zeros((n_samples, self.k + 1))
            for idx in range(self.k + 1):
                S[arr[idx], idx] = 1
            return S

        n_samples = X.shape[0]

        if self.c > n_samples:
            getLogger(__name__).error(
                "Cannot find %d clusters with n_samples = %d",
                self.c, n_samples)
            raise ValueError(
                "Cannot find %d clusters with n_samples = %d"
                % (self.c, n_samples))

        if self.k >= n_samples:
            getLogger(__name__).error(
                "Cannot select %d nearest neighbors with n_samples = %d",
                self.k, n_samples)
            raise ValueError(
                "Cannot select %d nearest neighbors with n_samples = %d"
                % (self.k, n_samples))

        indices = list(range(n_samples))
        I = np.eye(self.k + 1)
        H = I - np.ones((self.k + 1, self.k + 1)) / (self.k + 1)

        neighbors = NearestNeighbors(
            n_neighbors=self.k + 1,
            algorithm='ball_tree').fit(X).kneighbors(X, return_distance=False)
        getLogger(__name__).info("Neighbors graph: %s", neighbors)
        X_centered = np.apply_along_axis(
            lambda arr: X[arr].T.dot(H), 1, neighbors)

        S = np.apply_along_axis(lambda arr: construct_S(arr), 1, neighbors)
        getLogger(__name__).info("S: %s", S)
        B = np.vectorize(
            lambda idx: np.linalg.inv(X_centered[idx].T.dot(X_centered[idx])
                        + self.l * I),
            signature='()->(1,1)')(indices)
        getLogger(__name__).info("B: %s", B)
        Mi = np.vectorize(
            lambda idx: S[idx].dot(H).dot(B[idx]).dot(H).dot(S[idx].T),
            signature='()->(1,1)')(indices)
        M = X.T.dot(Mi.sum(axis=0)).dot(X)
        getLogger(__name__).info("M: %s", M)

        D = np.eye(self.n_features_)
        previous_target = -1
        for step in range(self.max_iterations):
            P = M + self.gamma * D
            getLogger(__name__).info("P: %s", P)
            _, W = eigh(a=P, subset_by_index=[0, self.c - 1])
            getLogger(__name__).info("W: %s", W)
            diag = 2 * matrix_norm(W)
            diag[diag < 1e-10] = 1e-10  # prevents division by zero
            D = np.diag(1 / diag)
            getLogger(__name__).info("D: %s", D)

            target = np.trace(W.T.dot(M).dot(W)) + self.gamma * l21_norm(W)
            getLogger(__name__).info("New target value: %d", target)
            if abs(target - previous_target) < self.epsilon:
                break
            previous_target = target

        getLogger(__name__).info("Ended up with W = %s", W)
        self.feature_scores_ = matrix_norm(W)
        getLogger(__name__).info("Feature scores: %s", self.feature_scores_)
        ranking = np.argsort(self.feature_scores_)[::-1]
        self.selected_features_ = ranking[:self.n_features]


###########################################################################
# from ...utils.information_theory import *


def MIM(selected_features, free_features, x, y, **kwargs):
    """Mutual Information Maximization feature scoring criterion. This
    criterion focuses only on increase of relevance. Given set of already
    selected features and set of remaining features on dataset X with labels
    y selects next feature.

    Parameters
    ----------
    selected_features : list of ints
        already selected features
    free_features : list of ints
        free features
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.
    kwargs : dict, optional
        Additional parameters to pass to generalizedCriteria.

    Returns
    -------
    array-like, shape (n_features,) : feature scores
        
    Notes
    -----
    For more details see `this paper
    <http://www.jmlr.org/papers/volume13/brown12a/brown12a.pdf/>`_.

    Examples
    --------
    >>> from ITMO_FS.filters.multivariate import MIM
    >>> from sklearn.preprocessing import KBinsDiscretizer
    >>> import numpy as np
    >>> x = np.array([[1, 2, 3, 3, 1], [2, 2, 3, 3, 2], [1, 3, 3, 1, 3],
    ... [3, 1, 3, 1, 4], [4, 4, 3, 1, 5]])
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
    >>> x = est.fit_transform(x)
    >>> selected_features = [1, 2]
    >>> other_features = [i for i in range(0, x.shape[1]) if i
    ... not in selected_features]
    >>> MIM(np.array(selected_features), np.array(other_features), x, y)
    array([1.33217904, 0.67301167, 1.60943791])
    """
    return matrix_mutual_information(x[:, free_features], y)


def MRMR(selected_features, free_features, x, y, **kwargs):
    """Minimum-Redundancy Maximum-Relevance feature scoring criterion. Given
    set of already selected features and set of remaining features on
    dataset X with labels y selects next feature.

    Parameters
    ----------
    selected_features : list of ints
        already selected features
    free_features : list of ints
        free features
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.
    kwargs : dict, optional
        Additional parameters to pass to generalizedCriteria.

    Returns
    -------
    array-like, shape (n_features,) : feature scores

    Notes
    -----
    For more details see `this paper
    <http://www.jmlr.org/papers/volume13/brown12a/brown12a.pdf/>`_.

    Examples
    --------
    >>> from ITMO_FS.filters.multivariate import MRMR
    >>> from sklearn.preprocessing import KBinsDiscretizer
    >>> import numpy as np
    >>> x = np.array([[1, 2, 3, 3, 1], [2, 2, 3, 3, 2], [1, 3, 3, 1, 3],
    ... [3, 1, 3, 1, 4], [4, 4, 3, 1, 5]])
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
    >>> x = est.fit_transform(x)
    >>> selected_features = []
    >>> other_features = [i for i in range(0, x.shape[1]) if i
    ... not in selected_features]
    >>> MRMR(np.array(selected_features), np.array(other_features), x, y)
    array([1.33217904, 1.33217904, 0.        , 0.67301167, 1.60943791])
    >>> selected_features = [1, 2]
    >>> other_features = [i for i in range(0, x.shape[1]) if i
    ... not in selected_features]
    >>> MRMR(np.array(selected_features), np.array(other_features), x, y)
    array([0.80471896, 0.33650583, 0.94334839])
    """
    if selected_features.size == 0:
        return matrix_mutual_information(x, y)
    return generalizedCriteria(
        selected_features, free_features, x, y, 1 / selected_features.size, 0,
        **kwargs)


def JMI(selected_features, free_features, x, y, **kwargs):
    """Joint Mutual Information feature scoring criterion. Given set of already
    selected features and set of remaining features on dataset X with labels
    y selects next feature.

    Parameters
    ----------
    selected_features : list of ints
        already selected features
    free_features : list of ints
        free features
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.
    kwargs : dict, optional
        Additional parameters to pass to generalizedCriteria.

    Returns
    -------
    array-like, shape (n_features,) : feature scores
        
    Notes
    -----
    For more details see `this paper
    <http://www.jmlr.org/papers/volume13/brown12a/brown12a.pdf/>`_.

    Examples
    --------
    >>> from ITMO_FS.filters.multivariate import JMI
    >>> from sklearn.preprocessing import KBinsDiscretizer
    >>> import numpy as np
    >>> x = np.array([[1, 2, 3, 3, 1], [2, 2, 3, 3, 2], [1, 3, 3, 1, 3],
    ... [3, 1, 3, 1, 4], [4, 4, 3, 1, 5]])
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
    >>> x = est.fit_transform(x)
    >>> selected_features = []
    >>> other_features = [i for i in range(0, x.shape[1]) if i
    ... not in selected_features]
    >>> JMI(np.array(selected_features), np.array(other_features), x, y)
    array([1.33217904, 1.33217904, 0.        , 0.67301167, 1.60943791])
    >>> selected_features = [1, 2]
    >>> other_features = [i for i in range(0, x.shape[1]) if i
    ... not in selected_features]
    >>> JMI(np.array(selected_features), np.array(other_features), x, y)
    array([0.80471896, 0.33650583, 0.94334839])
    """
    if selected_features.size == 0:
        return matrix_mutual_information(x, y)
    return generalizedCriteria(
        selected_features, free_features, x, y, 1 / selected_features.size,
        1 / selected_features.size, **kwargs)

def JMIM(selected_features, free_features, x, y, **kwargs):
    """Joint Mutual Information Maximization feature scoring criterion. Given
    set of already selected features and set of remaining features on
    dataset X with labels y selects next feature.

    Parameters
    ----------
    selected_features : list of ints
        already selected features
    free_features : list of ints
        free features
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.
    kwargs : dict, optional
        Additional parameters to pass to generalizedCriteria.

    Returns
    -------
    array-like, shape (n_features,) : feature scores

    Notes
    -----
    For more details see `this paper
    <https://www.sciencedirect.com/science/article/pii/S0957417415004674/>`_.

    Examples
    --------
    >>> from ITMO_FS.filters.multivariate import JMIM
    >>> from sklearn.preprocessing import KBinsDiscretizer
    >>> import numpy as np
    >>> x = np.array([[1, 2, 3, 3, 1], [2, 2, 3, 3, 2], [1, 3, 3, 1, 3],
    ... [3, 1, 3, 1, 4], [4, 4, 3, 1, 5]])
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
    >>> x = est.fit_transform(x)
    >>> selected_features = []
    >>> other_features = [i for i in range(0, x.shape[1]) if i
    ... not in selected_features]
    >>> JMIM(np.array(selected_features), np.array(other_features), x, y)
    array([1.33217904, 1.33217904, 0.        , 0.67301167, 1.60943791])
    >>> selected_features = [1, 2]
    >>> other_features = [i for i in range(0, x.shape[1]) if i
    ... not in selected_features]
    >>> JMIM(np.array(selected_features), np.array(other_features), x, y)
    array([1.33217904, 0.67301167, 1.60943791])
    """
    relevance = matrix_mutual_information(x[:, free_features], y)
    if selected_features.size == 0:
        return relevance
    cond_information = np.vectorize(
        lambda free_feature: np.apply_along_axis(
            conditional_mutual_information, 0, x[:, selected_features],
            y, x[:, free_feature]),
        signature='()->(1)')(free_features)
    return np.min(cond_information.T + relevance, axis=0)

def NJMIM(selected_features, free_features, x, y, **kwargs):
    """Normalized Joint Mutual Information Maximization feature scoring
    criterion. Given set of already selected features and set of
    remaining features on dataset X with labels y selects next feature.

    Parameters
    ----------
    selected_features : list of ints
        already selected features
    free_features : list of ints
        free features
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.
    kwargs : dict, optional
        Additional parameters to pass to generalizedCriteria.

    Returns
    -------
    array-like, shape (n_features,) : feature scores

    Notes
    -----
    For more details see `this paper
    <https://www.sciencedirect.com/science/article/pii/S0957417415004674/>`_.

    Examples
    --------
    >>> from ITMO_FS.filters.multivariate import NJMIM
    >>> from sklearn.preprocessing import KBinsDiscretizer
    >>> import numpy as np
    >>> x = np.array([[1, 2, 3, 3, 1], [2, 2, 3, 3, 2], [1, 3, 3, 1, 3],
    ... [3, 1, 3, 1, 4], [4, 4, 3, 1, 5]])
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
    >>> x = est.fit_transform(x)
    >>> selected_features = []
    >>> other_features = [i for i in range(0, x.shape[1]) if i
    ... not in selected_features]
    >>> NJMIM(np.array(selected_features), np.array(other_features), x, y)
    array([1.33217904, 1.33217904, 0.        , 0.67301167, 1.60943791])
    >>> selected_features = [1, 2]
    >>> other_features = [i for i in range(0, x.shape[1]) if i
    ... not in selected_features]
    >>> NJMIM(np.array(selected_features), np.array(other_features), x, y)
    array([0.82772938, 0.41816566, 1.        ])
    """
    if selected_features.size == 0:
        return matrix_mutual_information(x, y)
    sym_relevance = np.vectorize(
        lambda selected_feature: np.apply_along_axis(
            symmetrical_relevance, 0, x[:, free_features],
            x[:, selected_feature], y),
        signature='()->(1)')(selected_features)
    return np.min(sym_relevance, axis=0)


def CIFE(selected_features, free_features, x, y, **kwargs):
    """Conditional Infomax Feature Extraction feature scoring criterion. Given
    set of already selected features and set of remaining features on
    dataset X with labels y selects next feature.

    Parameters
    ----------
    selected_features : list of ints
        already selected features
    free_features : list of ints
        free features
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.
    kwargs : dict, optional
        Additional parameters to pass to generalizedCriteria.

    Returns
    -------
    array-like, shape (n_features,) : feature scores
        
    Notes
    -----
    For more details see `this paper
    <http://www.jmlr.org/papers/volume13/brown12a/brown12a.pdf/>`_.

    Examples
    --------
    >>> from ITMO_FS.filters.multivariate import CIFE
    >>> from sklearn.preprocessing import KBinsDiscretizer
    >>> import numpy as np
    >>> x = np.array([[1, 2, 3, 3, 1], [2, 2, 3, 3, 2], [1, 3, 3, 1, 3],
    ... [3, 1, 3, 1, 4], [4, 4, 3, 1, 5]])
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
    >>> x = est.fit_transform(x)
    >>> selected_features = []
    >>> other_features = [i for i in range(0, x.shape[1]) if i
    ... not in selected_features]
    >>> CIFE(np.array(selected_features), np.array(other_features), x, y)
    array([1.33217904, 1.33217904, 0.        , 0.67301167, 1.60943791])
    >>> selected_features = [1, 2]
    >>> other_features = [i for i in range(0, x.shape[1]) if i
    ... not in selected_features]
    >>> CIFE(np.array(selected_features), np.array(other_features), x, y)
    array([0.27725887, 0.        , 0.27725887])
    """
    return generalizedCriteria(
        selected_features, free_features, x, y, 1, 1, **kwargs)


def MIFS(selected_features, free_features, x, y, beta, **kwargs):
    """Mutual Information Feature Selection feature scoring criterion. This
    criterion includes the I(X;Y) term to ensure feature relevance,
    but introduces a penalty to enforce low correlations with features
    already selected in set. Given set of already selected features and set
    of remaining features on dataset X with labels y selects next feature.

    Parameters
    ----------
    selected_features : list of ints
        already selected features
    free_features : list of ints
        free features
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.
    beta : float
        Coefficient for redundancy term.
    kwargs : dict, optional
        Additional parameters to pass to generalizedCriteria.

    Returns
    -------
    array-like, shape (n_features,) : feature scores
        
    Notes
    -----
    For more details see `this paper
    <http://www.jmlr.org/papers/volume13/brown12a/brown12a.pdf/>`_.

    Examples
    --------
    >>> from ITMO_FS.filters.multivariate import MIFS
    >>> from sklearn.preprocessing import KBinsDiscretizer
    >>> import numpy as np
    >>> x = np.array([[1, 2, 3, 3, 1], [2, 2, 3, 3, 2], [1, 3, 3, 1, 3],
    ... [3, 1, 3, 1, 4], [4, 4, 3, 1, 5]])
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
    >>> x = est.fit_transform(x)
    >>> selected_features = []
    >>> other_features = [i for i in range(0, x.shape[1]) if i
    ... not in selected_features]
    >>> MIFS(np.array(selected_features), np.array(other_features), x, y, 0.4)
    array([1.33217904, 1.33217904, 0.        , 0.67301167, 1.60943791])
    >>> selected_features = [1, 2]
    >>> other_features = [i for i in range(0, x.shape[1]) if i
    ... not in selected_features]
    >>> MIFS(np.array(selected_features), np.array(other_features), x, y, 0.4)
    array([0.91021097, 0.403807  , 1.0765663 ])
    """
    return generalizedCriteria(
        selected_features, free_features, x, y, beta, 0, **kwargs)


def CMIM(selected_features, free_features, x, y, **kwargs):
    """Conditional Mutual Info Maximisation feature scoring criterion. Given
    set of already selected features and set of remaining features on
    dataset X with labels y selects next feature.

    Parameters
    ----------
    selected_features : list of ints
        already selected features
    free_features : list of ints
        free features
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.
    kwargs : dict, optional
        Additional parameters to pass to generalizedCriteria.

    Returns
    -------
    array-like, shape (n_features,) : feature scores
        
    Notes
    -----
    For more details see `this paper
    <http://www.jmlr.org/papers/volume13/brown12a/brown12a.pdf/>`_.

    Examples
    --------
    >>> from ITMO_FS.filters.multivariate import CMIM
    >>> from sklearn.preprocessing import KBinsDiscretizer
    >>> import numpy as np
    >>> x = np.array([[1, 2, 3, 3, 1], [2, 2, 3, 3, 2], [1, 3, 3, 1, 3],
    ... [3, 1, 3, 1, 4], [4, 4, 3, 1, 5]])
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
    >>> x = est.fit_transform(x)
    >>> selected_features = []
    >>> other_features = [i for i in range(0, x.shape[1]) if i
    ... not in selected_features]
    >>> CMIM(np.array(selected_features), np.array(other_features), x, y)
    array([1.33217904, 1.33217904, 0.        , 0.67301167, 1.60943791])
    >>> selected_features = [1, 2]
    >>> other_features = [i for i in range(0, x.shape[1]) if i
    ... not in selected_features]
    >>> CMIM(np.array(selected_features), np.array(other_features), x, y)
    array([0.27725887, 0.        , 0.27725887])
    """
    if selected_features.size == 0:
        return matrix_mutual_information(x, y)
    vectorized_function = lambda free_feature: min(
        np.vectorize(
            lambda selected_feature: conditional_mutual_information(
                x[:, free_feature], y,
                x[:, selected_feature]))(selected_features))
    return np.vectorize(vectorized_function)(free_features)


def ICAP(selected_features, free_features, x, y, **kwargs):
    """Interaction Capping feature scoring criterion. Given set of already
    selected features and set of remaining features on dataset X with labels
    y selects next feature.

    Parameters
    ----------
    selected_features : list of ints
        already selected features
    free_features : list of ints
        free features
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.
    kwargs : dict, optional
        Additional parameters to pass to generalizedCriteria.

    Returns
    -------
    array-like, shape (n_features,) : feature scores
        
    Notes
    -----
    For more details see `this paper
    <http://www.jmlr.org/papers/volume13/brown12a/brown12a.pdf/>`_.

    Examples
    --------
    >>> from ITMO_FS.filters.multivariate import ICAP
    >>> from sklearn.preprocessing import KBinsDiscretizer
    >>> import numpy as np
    >>> x = np.array([[1, 2, 3, 3, 1], [2, 2, 3, 3, 2], [1, 3, 3, 1, 3],
    ... [3, 1, 3, 1, 4], [4, 4, 3, 1, 5]])
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
    >>> x = est.fit_transform(x)
    >>> selected_features = []
    >>> other_features = [i for i in range(0, x.shape[1]) if i
    ... not in selected_features]
    >>> ICAP(np.array(selected_features), np.array(other_features), x, y)
    array([1.33217904, 1.33217904, 0.        , 0.67301167, 1.60943791])
    >>> selected_features = [1, 2]
    >>> other_features = [i for i in range(0, x.shape[1]) if i
    ... not in selected_features]
    >>> ICAP(np.array(selected_features), np.array(other_features), x, y)
    array([0.27725887, 0.        , 0.27725887])
    """
    if "relevance" in kwargs:
        relevance = kwargs["relevance"]
    else:
        relevance = matrix_mutual_information(x[:, free_features], y)

    if selected_features.size == 0:
        return relevance

    redundancy = np.vectorize(
        lambda free_feature: matrix_mutual_information(
            x[:, selected_features], 
            x[:, free_feature]),
        signature='()->(1)')(free_features)
    cond_dependency = np.vectorize(
        lambda free_feature: np.apply_along_axis(
            conditional_mutual_information, 0,
            x[:, selected_features],
            x[:, free_feature], y),
        signature='()->(1)')(free_features)
    return relevance - np.sum(
        np.maximum(redundancy - cond_dependency, 0.), axis=1)


def DCSF(selected_features, free_features, x, y, **kwargs):
    """Dynamic change of selected feature with the class scoring criterion.
    DCSF employs both mutual information and conditional mutual information
    to find an optimal subset of features. Given set of already selected
    features and set of remaining features on dataset X with labels y
    selects next feature.

    Parameters
    ----------
    selected_features : list of ints
        already selected features
    free_features : list of ints
        free features
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.
    kwargs : dict, optional
        Additional parameters to pass to generalizedCriteria.

    Returns
    -------
    array-like, shape (n_features,) : feature scores
        
    Notes
    -----
    For more details see `this paper
    <https://www.sciencedirect.com/science/article/abs/pii/S0031320318300736/>`_.
        
    Examples
    --------
    >>> from ITMO_FS.filters.multivariate import DCSF
    >>> from sklearn.preprocessing import KBinsDiscretizer
    >>> import numpy as np
    >>> x = np.array([[1, 2, 3, 3, 1], [2, 2, 3, 3, 2], [1, 3, 3, 1, 3],
    ... [3, 1, 3, 1, 4], [4, 4, 3, 1, 5]])
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
    >>> x = est.fit_transform(x)
    >>> selected_features = []
    >>> other_features = [i for i in range(0, x.shape[1]) if i
    ... not in selected_features]
    >>> DCSF(np.array(selected_features), np.array(other_features), x, y)
    array([0., 0., 0., 0., 0.])
    >>> selected_features = [1, 2]
    >>> other_features = [i for i in range(0, x.shape[1]) if i
    ... not in selected_features]
    >>> DCSF(np.array(selected_features), np.array(other_features), x, y)
    array([0.83177662, 0.65916737, 0.55451774])
    """
    if selected_features.size == 0:
        return np.zeros(len(free_features))
    vectorized_function = lambda free_feature: np.sum(
        np.apply_along_axis(
            lambda z, a, b: conditional_mutual_information(a, b, z), 0,
            x[:, selected_features],
            x[:, free_feature], y)
        + np.apply_along_axis(
            conditional_mutual_information, 0, x[:, selected_features], y,
            x[:, free_feature])
        - matrix_mutual_information(
            x[:, selected_features], x[:, free_feature]))
    return np.vectorize(vectorized_function)(free_features)


def CFR(selected_features, free_features, x, y, **kwargs):
    """The criterion of CFR maximizes the correlation and minimizes the
    redundancy. Given set of already selected features and set of remaining
    features on dataset X with labels y selects next feature.

    Parameters
    ----------
    selected_features : list of ints
        already selected features
    free_features : list of ints
        free features
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.
    kwargs : dict, optional
        Additional parameters to pass to generalizedCriteria.

    Returns
    -------
    array-like, shape (n_features,) : feature scores
        
    Notes
    -----
    For more details see `this paper
    <https://www.sciencedirect.com/science/article/pii/S2210832719302522/>`_.

    Examples
    --------
    >>> from ITMO_FS.filters.multivariate import CFR
    >>> from sklearn.preprocessing import KBinsDiscretizer
    >>> import numpy as np
    >>> x = np.array([[1, 2, 3, 3, 1], [2, 2, 3, 3, 2], [1, 3, 3, 1, 3],
    ... [3, 1, 3, 1, 4], [4, 4, 3, 1, 5]])
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
    >>> x = est.fit_transform(x)
    >>> selected_features = []
    >>> other_features = [i for i in range(0, x.shape[1]) if i
    ... not in selected_features]
    >>> CFR(np.array(selected_features), np.array(other_features), x, y)
    array([0., 0., 0., 0., 0.])
    >>> selected_features = [1, 2]
    >>> other_features = [i for i in range(0, x.shape[1]) if i
    ... not in selected_features]
    >>> CFR(np.array(selected_features), np.array(other_features), x, y)
    array([0.55451774, 0.        , 0.55451774])
    """
    if selected_features.size == 0:
        return np.zeros(len(free_features))
    vectorized_function = lambda free_feature: np.sum(
        np.apply_along_axis(
            lambda z, a, b: conditional_mutual_information(a, b, z), 0,
            x[:, selected_features],
            x[:, free_feature], y)
        + np.apply_along_axis(
            conditional_mutual_information, 0, x[:, selected_features],
            x[:, free_feature], y)
        - matrix_mutual_information(
            x[:, selected_features], x[:, free_feature]))
    return np.vectorize(vectorized_function)(free_features)


def MRI(selected_features, free_features, x, y, **kwargs):
    """Max-Relevance and Max-Independence feature scoring criteria. Given set
    of already selected features and set of remaining features on dataset X
    with labels y selects next feature.

    Parameters
    ----------
    selected_features : list of ints
        already selected features
    free_features : list of ints
        free features
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.
    kwargs : dict, optional
        Additional parameters to pass to generalizedCriteria.

    Returns
    -------
    array-like, shape (n_features,) : feature scores
        
    Notes
    -----
    For more details see `this paper
    <https://link.springer.com/article/10.1007/s10489-019-01597-z/>`_.

    Examples
    --------
    >>> from ITMO_FS.filters.multivariate import MRI
    >>> from sklearn.preprocessing import KBinsDiscretizer
    >>> import numpy as np
    >>> x = np.array([[1, 2, 3, 3, 1], [2, 2, 3, 3, 2], [1, 3, 3, 1, 3],
    ... [3, 1, 3, 1, 4], [4, 4, 3, 1, 5]])
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
    >>> x = est.fit_transform(x)
    >>> selected_features = []
    >>> other_features = [i for i in range(0, x.shape[1]) if i
    ... not in selected_features]
    >>> MRI(np.array(selected_features), np.array(other_features), x, y)
    array([1.33217904, 1.33217904, 0.        , 0.67301167, 1.60943791])
    >>> selected_features = [1, 2]
    >>> other_features = [i for i in range(0, x.shape[1]) if i
    ... not in selected_features]
    >>> MRI(np.array(selected_features), np.array(other_features), x, y)
    array([0.62889893, 0.22433722, 0.72131855])
    """
    return generalizedCriteria(
        selected_features, free_features, x, y,
        2 / (selected_features.size + 1), 2 / (selected_features.size + 1),
        **kwargs)


def __information_weight(xk, xj, y):
    return 1 + (joint_mutual_information(xk, xj, y)
                - mutual_information(xk, y)
                - mutual_information(xj, y)) / (entropy(xk) + entropy(xj))


def __SU(xk, xj):
    return 2 * mutual_information(xk, xj) / (entropy(xk) + entropy(xj))


def IWFS(selected_features, free_features, x, y, **kwargs):
    """Interaction Weight base feature scoring criteria. IWFS is good at
    identifyng Given set of already selected features and set of remaining
    features on dataset X with labels y selects next feature.

    Parameters
    ----------
    selected_features : list of ints
        already selected features
    free_features : list of ints
        free features
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.
    kwargs : dict, optional
        Additional parameters to pass to generalizedCriteria.

    Returns
    -------
    array-like, shape (n_features,) : feature scores
        
    Notes
    -----
    For more details see `this paper
    <https://www.sciencedirect.com/science/article/abs/pii/S0031320315000850/>`_.

    Examples
    --------
    >>> from ITMO_FS.filters.multivariate import IWFS
    >>> from sklearn.preprocessing import KBinsDiscretizer
    >>> import numpy as np
    >>> x = np.array([[1, 2, 3, 3, 1], [2, 2, 3, 3, 2], [1, 3, 3, 1, 3],
    ... [3, 1, 3, 1, 4], [4, 4, 3, 1, 5]])
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
    >>> x = est.fit_transform(x)
    >>> selected_features = []
    >>> other_features = [i for i in range(0, x.shape[1]) if i
    ... not in selected_features]
    >>> IWFS(np.array(selected_features), np.array(other_features), x, y)
    array([0., 0., 0., 0., 0.])
    >>> selected_features = [1, 2]
    >>> other_features = [i for i in range(0, x.shape[1]) if i
    ... not in selected_features]
    >>> IWFS(np.array(selected_features), np.array(other_features), x, y)
    array([1.0824043 , 1.11033338, 1.04268505])
    """
    if selected_features.size == 0:
        return np.zeros(len(free_features))
    vectorized_function = lambda free_feature: np.prod(
        np.apply_along_axis(
            lambda Xj, Xk, y: __information_weight(Xk, Xj, y),
            0, x[:, selected_features], x[:, free_feature], y)
        * (np.apply_along_axis(
            __SU, 0, x[:, selected_features], x[:, free_feature]) + 1))
    return np.vectorize(vectorized_function)(free_features)


# Ask question what should happen if number of features user want is less
# than useful number of features
def generalizedCriteria(selected_features, free_features, x, y, beta, gamma,
                        **kwargs):
    """This feature scoring criteria is a linear combination of all relevance,
    redundancy, conditional dependency Given set of already selected
    features and set of remaining features on dataset X with labels y
    selects next feature.

    Parameters
    ----------
    selected_features : list of ints
        already selected features
    free_features : list of ints
        free features
    x : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.
    beta : float
        Coefficient for redundancy term.
    gamma : float
        Coefficient for conditional dependancy term.

    Returns
    -------
    array-like, shape (n_features,) : feature scores

    Notes
    -----
    See the original paper [1]_ for more details.

    References
    ----------
    .. [1] Brown, Gavin et al. "Conditional
    Likelihood Maximisation: A Unifying Framework for Information
    Theoretic Feature Selection." JMLR 2012.
        
    Examples
    --------
    >>> from ITMO_FS.filters.multivariate import CFR
    >>> from sklearn.preprocessing import KBinsDiscretizer
    >>> import numpy as np
    >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
    >>> x = np.array([[1, 2, 3, 3, 1], [2, 2, 3, 3, 2], [1, 3, 3, 1, 3],
    ... [3, 1, 3, 1, 4], [4, 4, 3, 1, 5]])
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> x = est.fit_transform(x)
    >>> selected_features = []
    >>> other_features = [i for i in range(0, x.shape[1]) if i
    ... not in selected_features]
    >>> generalizedCriteria(np.array(selected_features),
    ... np.array(other_features), x, y, 0.4, 0.3)
    array([1.33217904, 1.33217904, 0.        , 0.67301167, 1.60943791])
    >>> selected_features = [1, 2]
    >>> other_features = [i for i in range(0, x.shape[1]) if i
    ... not in selected_features]
    >>> generalizedCriteria(np.array(selected_features),
    ... np.array(other_features), x, y, 0.4, 0.3)
    array([0.91021097, 0.403807  , 1.0765663 ])
    """
    if "relevance" in kwargs:
        relevance = kwargs["relevance"]
    else:
        relevance = matrix_mutual_information(x[:, free_features], y)

    if selected_features.size == 0:
        return relevance

    if beta != 0:
        if "redundancy" in kwargs:
            redundancy = kwargs["redundancy"]
        else:
            redundancy = np.vectorize(
                lambda free_feature: np.sum(
                    matrix_mutual_information(
                        x[:, selected_features],
                        x[:, free_feature])))(free_features)
    else:
        redundancy = 0

    if gamma != 0:
        cond_dependency = np.vectorize(
            lambda free_feature: np.sum(
                np.apply_along_axis(
                    conditional_mutual_information, 0, x[:, selected_features],
                    x[:, free_feature], y)))(free_features)
    else:
        cond_dependency = 0
    return relevance - beta*redundancy + gamma*cond_dependency


MEASURE_NAMES = {"MIM": MIM,
                 "MRMR": MRMR,
                 "JMI": JMI,
                 "CIFE": CIFE,
                 "MIFS": MIFS,
                 "CMIM": CMIM,
                 "ICAP": ICAP,
                 "DCSF": DCSF,
                 "CFR": CFR,
                 "MRI": MRI,
                 "IWFS": IWFS,
                 "generalizedCriteria": generalizedCriteria}


import numpy as np
import random
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from functools import partial
# from ...utils.information_theory import *
# from ...utils import BaseTransformer

# TODO fix docs


def genes_mutual_information(genes):
    """
    :param genes: dataset
    :return: mutual information for every gene in dataset
    """
    g_num, _ = genes.shape  # number of features
    mi_matrix = np.zeros((g_num, g_num))
    for i in range(g_num):
        for j in range(g_num):
            if i != j:
                mi_matrix[i][j] = mutual_information(genes[i], genes[j])
    mi_vector = [sum(mi_matrix[i]) for i in range(g_num)]
    return mi_vector


def decode_genes(mapping, chromosome, train, test):
    """
    :param chromosome: binary vector of feature presence
    :param train: train set of initial dataset
    :param test: test set of initial dataset
    :return: decoded train and test sets (reduced)
    """
    filtered_train, filtered_test = [], []
    for i in range(len(chromosome)):
        if chromosome[i] == 1:
            initial_index = mapping[i]
            filtered_train.append(train[initial_index])
            filtered_test.append(test[initial_index])
    return np.array(filtered_train), np.array(filtered_test)


def population_fitness(
        mapping,
        population,
        train,
        train_cl,
        test,
        test_cl,
        measure):
    """
    :param population: vector of chromosomes
    :return: vector of (chromosome code, chromosome fitness), max fitness, average fitness
    """
    code_fitness = []
    f_sum = 0
    for i in range(len(population)):
        filtered_train, filtered_test = decode_genes(
            mapping, population[i], train, test)
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        if len(filtered_train) == 0:
            continue
        clf.fit(filtered_train.transpose(), train_cl)
        predicted_classes = clf.predict(filtered_test.transpose())
        f = measure(test_cl, predicted_classes)
        code_fitness.append((population[i], f))
        f_sum += f
    code_fitness.sort(key=lambda p: p[1], reverse=True)
    f_max = code_fitness[0][1]
    f_avg = f_sum / len(population)
    return code_fitness, f_max, f_avg


def crossover(x, y):
    """ simple one-point crossover """
    random_point = random.randint(1, len(x) - 1)
    return x[0:random_point] + y[random_point:len(x)], \
        y[0:random_point] + x[random_point:len(x)]


def mutation(x):
    """ simple one-bit-inverse mutation """
    random_point = random.randint(0, len(x) - 1)
    x[random_point] = (x[random_point] - 1) % 2
    return x


def cross_and_mutate(pc, pm, population):
    """
    :param pc: crossover probability
    :param pm: mutation probability
    :param population: (chromosome code, chromosome fitness) pairs
    :return: (new population, maximum parents' fitness) pair
    """
    cross_number = int(pc * len(population))
    mutate_number = int(pm * len(population))
    max_parent_f = 0
    new_population = list(map(lambda x: x[0], population))
    for i in range(cross_number):
        parent1, f1 = population[random.randint(0, len(population) - 1)]
        parent2, f2 = population[random.randint(0, len(population) - 1)]
        child1, child2 = crossover(parent1, parent2)
        new_population.extend([child1, child2])
        max_parent_f = max([max_parent_f, f1, f2])
    for i in range(mutate_number):
        mutant = mutation(
            population[random.randint(0, len(population) - 1)][0])
        new_population.append(mutant)
    return new_population, max_parent_f


class MIMAGA(BaseTransformer):

    def _fit(self, X, y, param):
        pass

    def __init__(self, mim_size, pop_size, max_iter=20, f_target=0.8, k1=0.6,
                 k2=0.3, k3=0.9, k4=0.001):
        """
        :param mim_size: desirable number of filtered features after MIM
        :param pop_size: initial population size
        :param max_iter: maximum number of iterations in algorithm
        :param f_target: desirable fitness value
        :param k1: consts to determine crossover probability
        :param k2: consts to determine crossover probability
        :param k3: consts to determine mutation probability
        :param k4: consts to determine mutation probability

        See also
        --------
        https://www.sciencedirect.com/science/article/abs/pii/S0925231217304150
        """
        self.mim_size = mim_size
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.f_target = f_target
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4

    # MIM

    def _mim_filter(self, genes):
        """
        :param genes: initial dataset
        :return: sequence of feature indexes with minimum MI
        """
        g_num, _ = genes.shape
        mi_vector = genes_mutual_information(genes)
        seq_nums = [i for i in range(g_num)]
        target_sequence = list(map(lambda p: p[1], sorted(zip(mi_vector, seq_nums))))[
            :self.mim_size]
        return target_sequence

    # AGA
    def _initial_population(self):
        """
        :return: initial population
        P.S. each individual corresponds to chromosome
        """
        population = []
        for _ in range(self.pop_size):
            individual_num = random.randint(1, 2 << self.mim_size - 1)
            individual_code = list(
                map(int, bin(individual_num)[2:].zfill(self.mim_size)))
            population.append(individual_code)
        return population

    def _crossover_probability(self, f_max, f_avg, f_par):
        """ probability of crossover in population """
        if f_par >= f_avg:
            return self.k1 * ((f_max - f_par) / (f_max - f_avg)) \
                if f_max != f_avg else 1
        else:
            return self.k2

    def _mutation_probability(self, f_max, f_avg, f_par):
        """ probability of mutation in population """
        if f_par >= f_avg:
            return self.k3 * ((f_max - f_par) / (f_max - f_avg)) \
                if f_max != f_avg else 1
        else:
            return self.k4

    def _aga_filter(
            self,
            max_size,
            mapping,
            population,
            train,
            train_cl,
            test,
            test_cl):
        """
        :param max_size: maximum size of population (if population becomes bigger,
                         the worst individuals are killed)
        :param mapping: mapping from mim-filter index to initial index in dataset
        :param population: vector of chromosomes
        :param train: train set of initial dataset
        :param train_cl: class distribution of initial train dataset
        :param test: test set of initial dataset
        :param test_cl: class distribution of initial test dataset
        :return: best individual (sequence of features), it's fitness value
        """
        f_par = f_max = 0
        counter = 0
        best_individual = [1 for _ in range(len(population[0]))]
        while counter < self.max_iter and f_max < self.f_target:
            code_fitness, f_max, f_avg = population_fitness(
                mapping, population, train, train_cl, test, test_cl, partial(
                    f1_score, average='macro'))
            if len(code_fitness) > max_size:
                code_fitness = code_fitness[:max_size]
                population = list(map(lambda x: x[0], code_fitness))

            highly_fitted = list(
                filter(
                    lambda x: x[1] >= f_max / 2,
                    code_fitness))
            if len(highly_fitted) == 0:
                highly_fitted = code_fitness
            best_individual = code_fitness[0][0]

            pc = self._crossover_probability(f_max, f_avg, f_par)
            pm = self._mutation_probability(f_max, f_avg, f_par)
            new_generation, f_par = cross_and_mutate(pc, pm, highly_fitted)
            population = population + new_generation
            counter += 1
        return best_individual, f_max

    def mimaga_filter(self, genes, classes):
        """
        The main function to run algorithm
        :param genes: initial dataset in format: samples are rows, features are columns
        :param classes: distribution pf initial dataset
        :return: filtered with MIMAGA dataset, fitness value
        """
        train_set, test_set, train_classes, test_classes = train_test_split(
            genes, classes, test_size=0.33)
        filtered_indexes = self._mim_filter(train_set.transpose())
        index_map = dict(
            zip([i for i in range(self.mim_size)], filtered_indexes))

        first_population = self._initial_population()
        best, max_fitness = self._aga_filter(self.pop_size * 2, index_map, first_population,
                                             train_set.transpose(), train_classes, test_set.transpose(), test_classes)
        result_genes, _ = decode_genes(
            index_map, best, train_set.transpose(), test_set.transpose())
        return result_genes, max_fitness

# TODO: optimize everything bcs this works for hours
# mimaga = MIMAGA(30, 20, 20, 0.8, 0.6, 0.3, 0.9, 0.001)
# res_dataset, fitness = mimaga.mimaga_filter(dataset, distribution)


from logging import getLogger

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler

# from ...utils import knn_from_class, BaseTransformer


class STIR(BaseTransformer):
    """Feature selection using STIR algorithm.

    Parameters
    ----------
    n_features : int
        Number of features to select.
    metric : str or callable
        Distance metric to use in kNN. If str, should be one of the standard
        distance metrics (e.g. 'euclidean' or 'manhattan'). If callable, should
        have the signature metric(x1 (array-like, shape (n,)), x2 (array-like,
        shape (n,))) that should return the distance between two vectors.
    k : int
        Number of constant nearest hits/misses.

    Notes
    -----
    For more details see `this paper <https://academic.oup.com/bioinformatics/article/35/8/1358/5100883>`_.

    Examples
    --------
    >>> from ITMO_FS.filters.multivariate import STIR
    >>> import numpy as np
    >>> X = np.array([[3, 3, 3, 2, 2], [3, 3, 1, 2, 3], [1, 3, 5, 1, 1],
    ... [3, 1, 4, 3, 1], [3, 1, 2, 3, 1]])
    >>> y = np.array([1, 2, 2, 1, 2])
    >>> model = STIR(2).fit(X, y)
    >>> model.selected_features_
    array([2, 0], dtype=int64)
    """
    def __init__(self, n_features, metric='manhattan', k=1):
        self.n_features = n_features
        self.metric = metric
        self.k = k

    def _fit(self, X, y):
        """Fit the filter.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        y : array-like, shape (n_samples,)
            The classes for the samples.

        Returns
        -------
        None
        """
        n_samples = X.shape[0]
        classes, counts = np.unique(y, return_counts=True)

        if np.any(counts <= self.k):
            getLogger(__name__).error(
                "Cannot select %d nearest neighbors because one of the classes "
                "has less than %d samples", self.k, self.k + 1)
            raise ValueError(
                "Cannot select %d nearest neighbors because one of the classes "
                "has less than %d samples" % (self.k, self.k + 1))

        x_normalized = MinMaxScaler().fit_transform(X)
        dm = pairwise_distances(x_normalized, x_normalized, self.metric)
        getLogger(__name__).info("Distance matrix: %s", dm)

        indices = np.arange(n_samples)
        hits_diffs = np.abs(
            np.vectorize(
                lambda index: (
                    x_normalized[index]
                    - x_normalized[knn_from_class(
                        dm, y, index, self.k, y[index])]),
                signature='()->(n,m)')(indices))
        getLogger(__name__).info("Hit differences matrix: %s", hits_diffs)
        misses_diffs = np.abs(
            np.vectorize(
                lambda index: (
                    x_normalized[index]
                    - x_normalized[knn_from_class(
                        dm, y, index, self.k, y[index], anyOtherClass=True)]),
                signature='()->(n,m)')(indices))
        getLogger(__name__).info("Miss differences matrix: %s", misses_diffs)

        H = np.mean(hits_diffs, axis=(0,1))
        getLogger(__name__).info("H: %s", H)
        M = np.mean(misses_diffs, axis=(0,1))
        getLogger(__name__).info("M: %s", M)
        var_H = np.var(hits_diffs, axis=(0,1))
        var_M = np.var(misses_diffs, axis=(0,1))

        # the 1 / (1 / |M| + 1 / |H|) ^ (1/2) multiplier is constant, we omit it
        self.feature_scores_ = (
            (M - H) * np.sqrt(2 * self.k * n_samples - 2)
            / (np.sqrt((self.k * n_samples - 1) * (var_H + var_M)) + 1e-15))
        getLogger(__name__).info("Feature scores: %s", self.feature_scores_)
        self.selected_features_ = np.argsort(self.feature_scores_)[::-1][
            :self.n_features]



from logging import getLogger

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

# from ...utils import BaseTransformer, generate_features

class TraceRatioFisher(BaseTransformer):
    """Creates TraceRatio(similarity based) feature selection filter
    performed in supervised way, i.e. fisher version

    Parameters
    ----------
    n_features : int
        Number of features to select.
    epsilon : float
        Lambda change threshold.

    Notes
    -----
    For more details see `this paper
    <https://www.aaai.org/Papers/AAAI/2008/AAAI08-107.pdf/>`_.

    Examples
    --------
    >>> from ITMO_FS.filters.multivariate import TraceRatioFisher
    >>> x = np.array([[1, 2, 3, 3, 1], [2, 2, 3, 3, 2], [1, 3, 3, 1, 3],
    ... [3, 1, 3, 1, 4], [4, 4, 3, 1, 5]])
    >>> y = np.array([1, 2, 1, 1, 2])
    >>> tracer = TraceRatioFisher(3).fit(x, y)
    >>> tracer.selected_features_
    array([0, 1, 3], dtype=int64)
    """
    def __init__(self, n_features, epsilon=1e-3):
        self.n_features = n_features
        self.epsilon = epsilon

    def _fit(self, X, y):
        """Fit the filter.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples
        y : array-like, shape (n_samples,)
            The target values

        Returns
        -------
        None
        """
        n_samples = X.shape[0]
        classes, counts = np.unique(y, return_counts=True)
        counts_d = {cl: counts[idx] for idx, cl in enumerate(classes)}
        getLogger(__name__).info("Class counts: %s", counts_d)

        A_within = pairwise_distances(
            y.reshape(-1, 1), metric=lambda x, y: (
                (x[0] == y[0]) / counts_d[x[0]]))
        L_within = np.eye(n_samples) - A_within
        getLogger(__name__).info("A_w: %s", A_within)
        getLogger(__name__).info("L_w: %s", L_within)

        L_between = A_within - np.ones((n_samples, n_samples)) / n_samples
        getLogger(__name__).info("L_b: %s", L_between)

        E = X.T.dot(L_within).dot(X)
        B = X.T.dot(L_between).dot(X)

        # we need only diagonal elements for trace calculation
        e = np.array(np.diag(E))
        b = np.array(np.diag(B))
        getLogger(__name__).info("E: %s", e)
        getLogger(__name__).info("B: %s", b)
        lam = 0
        prev_lam = -1
        while (lam - prev_lam >= self.epsilon):  # TODO: optimize
            score = b - lam * e
            getLogger(__name__).info("Score: %s", score)
            self.selected_features_ = np.argsort(score)[::-1][:self.n_features]
            getLogger(__name__).info(
                "New selected set: %s", self.selected_features_)
            prev_lam = lam
            lam = (np.sum(b[self.selected_features_])
                   / np.sum(e[self.selected_features_]))
            getLogger(__name__).info("New lambda: %d", lam)
        self.score_ = score
        self.lam_ = lam


from logging import getLogger

import numpy as np
from sklearn.metrics import pairwise_distances

# from ...utils import BaseTransformer, generate_features
# from ...utils.information_theory import (entropy, joint_entropy,
#                                          mutual_information)


def _complementarity(x_i, x_j, y):
    return (entropy(x_i) + entropy(x_j) + entropy(y) - joint_entropy(x_i, x_j)
            - joint_entropy(x_i, y) - joint_entropy(x_j, y)
            + joint_entropy(x_i, x_j, y))


def _chained_information(x_i, x_j, y):
    return (mutual_information(x_i, y) + mutual_information(x_j, y)
            + _complementarity(x_i, x_j, y))


class DISRWithMassive(BaseTransformer):
    """Create DISR (Double Input Symmetric Relevance) feature selection filter
    based on kASSI criterin for feature selection which aims at maximizing the
    mutual information avoiding, meanwhile, large multivariate density
    estimation. Its a kASSI criterion with approximation of the information of
    a set of variables by counting average information of subset on combination
    of two features. This formulation thus deals with feature complementarity
    up to order two by preserving the same computational complexity of the
    MRMR and CMIM criteria The DISR calculation is done using graph based
    solution.

    Parameters
    ----------
    n_features : int
        Number of features to select.

    Notes
    -----
    For more details see `this paper
    <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.318.6576&rep=rep1&type=pdf/>`_.

    Examples
    --------
    >>> from ITMO_FS.filters.multivariate import DISRWithMassive
    >>> import numpy as np
    >>> X = np.array([[1, 2, 3, 3, 1], [2, 2, 3, 3, 2], [1, 3, 3, 1, 3],
    ... [3, 1, 3, 1, 4], [4, 4, 3, 1, 5]])
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> disr = DISRWithMassive(3).fit(X, y)
    >>> disr.selected_features_
    array([0, 1, 4], dtype=int64)
    """
    def __init__(self, n_features):
        self.n_features = n_features

    def _fit(self, x, y):
        """Fit the filter.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values.

        Returns
        -------
        None
        """
        free_features = np.array([], dtype='int')
        self.selected_features_ = generate_features(x)
        self._edges = pairwise_distances(
            x.T, x.T, lambda xi, xj: (_chained_information(xi, xj, y)
                                      / (joint_entropy(xi, xj) + 1e-15)))
        np.fill_diagonal(self._edges, 0)
        getLogger(__name__).info("Graph weights: %s", self._edges)
    
        while len(self.selected_features_) != self.n_features:
            min_index = np.argmin(
                np.sum(self._edges[np.ix_(self.selected_features_,
                                          self.selected_features_)], axis=0))
            getLogger(__name__).info(
                "Removing feature %d from selected set",
                self.selected_features_[min_index])
            free_features = np.append(
                free_features, self.selected_features_[min_index])
            self.selected_features_ = np.delete(
                self.selected_features_, min_index)

        getLogger(__name__).info(
            "Selected set: %s, free set: %s", self.selected_features_,
            free_features)

        while True:
            selected_weights = np.sum(
                self._edges[np.ix_(self.selected_features_,
                                   self.selected_features_)], axis=0)
            getLogger(__name__).info(
                "Graph of selected set: %s", selected_weights)

            free_weights = np.sum(self._edges[np.ix_(self.selected_features_,
                                                     free_features)], axis=0)
            getLogger(__name__).info(
                "Free weights that would be added: %s", free_weights)

            difference = (
                free_weights.reshape(-1, 1)
                - self._edges[np.ix_(free_features, self.selected_features_)]
                - selected_weights)
            getLogger(__name__).info("Difference matrix: %s", difference)

            if np.all(difference <= 0):
                getLogger(__name__).info(
                    "All differences are non-positive, terminating")
                break
            index_add, index_del = np.unravel_index(
                np.argmax(difference), difference.shape)
            getLogger(__name__).info(
                "Maximum difference found at index (%d, %d), swapping those "
                "features", index_add, index_del)

            self.selected_features_[index_del], free_features[index_add] = (
                free_features[index_add], self.selected_features_[index_del])


from logging import getLogger

import numpy as np

# from ...utils import BaseTransformer, generate_features
# from ...utils.information_theory import entropy, conditional_entropy


class FCBFDiscreteFilter(BaseTransformer):
    """Create FCBF (Fast Correlation Based filter) feature selection filter
    based on mutual information criteria for data with discrete features. This
    filter finds best set of features by searching for a feature, which
    provides the most information about classification problem on given dataset
    at each step and then eliminating features which are less relevant than
    redundant.

    Parameters
    ----------
    delta : float
        Symmetric uncertainty value threshold.

    Notes
    -----
    For more details see `this paper
    <https://www.aaai.org/Papers/ICML/2003/ICML03-111.pdf/>`_.

    Examples
    --------
    >>> from ITMO_FS.filters.multivariate import FCBFDiscreteFilter
    >>> import numpy as np
    >>> X = np.array([[1, 2, 3, 3, 1], [2, 2, 3, 3, 2], [1, 3, 3, 1, 3],
    ... [3, 1, 3, 1, 4], [4, 4, 3, 1, 5]])
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> fcbf = FCBFDiscreteFilter().fit(X, y)
    >>> fcbf.selected_features_
    array([4], dtype=int64)
    """
    def __init__(self, delta=0.1):
        self.delta = delta

    def _fit(self, x, y):
        """Fit the filter.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values.

        Returns
        -------
        None
        """
        def __SU(x, y, entropy_y):
            entropy_x = entropy(x)
            return 2 * ((entropy_x - conditional_entropy(y, x))
                        / (entropy_x + entropy_y))

        free_features = generate_features(x)
        self.selected_features_ = np.array([], dtype='int')
        entropy_y = entropy(y)
        getLogger(__name__).info("Entropy of y: %f", entropy_y)

        su_class = np.apply_along_axis(__SU, 0, x, y, entropy_y)
        getLogger(__name__).info("SU values against y: %s", su_class)
        self.selected_features_ = np.argsort(su_class)[::-1][:
            np.count_nonzero(su_class > self.delta)]
        getLogger(__name__).info("Selected set: %s", self.selected_features_)

        index = 1
        while index < self.selected_features_.shape[0]:
            feature = self.selected_features_[index - 1]
            getLogger(__name__).info("Leading feature: %d", feature)
            entropy_feature = entropy(x[:, feature])
            getLogger(__name__).info(
                "Leading feature entropy: %f", entropy_feature)
            su_classes = su_class[self.selected_features_[index:]]
            getLogger(__name__).info(
                "SU values against y for the remaining features: %s",
                su_classes)
            su_feature = np.apply_along_axis(
                __SU, 0, x[:, self.selected_features_[index:]], x[:, feature],
                entropy_feature)
            getLogger(__name__).info(
                "SU values against leading feature for the remaining features: "
                "%s", su_feature)
            to_delete = np.flatnonzero(su_feature >= su_classes) + index
            getLogger(__name__).info(
                "Deleting those features from the selected set: %s",
                self.selected_features_[to_delete])
            self.selected_features_ = np.delete(
                self.selected_features_, to_delete)
            index += 1


from logging import getLogger

import numpy as np
from sklearn.base import TransformerMixin

# from .measures import (MEASURE_NAMES, mutual_information,
#                        matrix_mutual_information)
# from ...utils import BaseTransformer, generate_features


class MultivariateFilter(BaseTransformer):
    """Provides basic functionality for multivariate filters.

    Parameters
    ----------
    measure : string or callable
        A metric name defined in GLOB_MEASURE or a callable with signature
        measure(selected_features, free_features, dataset, labels) which
        should return a list of metric values for each feature in the dataset.
    n_features : int
        Number of features to select.
    beta : float, optional
        Initialize only in case you run MIFS or generalizedCriteria metrics.
    gamma : float, optional
        Initialize only in case you run generalizedCriteria metric.

    See Also
    --------

    Examples
    --------
    >>> from ITMO_FS.filters.multivariate import MultivariateFilter
    >>> from sklearn.preprocessing import KBinsDiscretizer
    >>> import numpy as np
    >>> est = KBinsDiscretizer(n_bins=10, encode='ordinal')
    >>> x = np.array([[1, 2, 3, 3, 1], [2, 2, 3, 3, 2], [1, 3, 3, 1, 3],
    ... [3, 1, 3, 1, 4], [4, 4, 3, 1, 5]])
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> data = est.fit_transform(x)
    >>> model = MultivariateFilter('JMI', 3).fit(x, y)
    >>> model.selected_features_
    array([4, 0, 1], dtype=int64)
    """
    def __init__(self, measure, n_features, beta=None, gamma=None):
        self.measure = measure
        self.n_features = n_features
        self.beta = beta
        self.gamma = gamma

    def _fit(self, X, y, **kwargs):
        """Fit the filter.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values.
        **kwargs

        Returns
        -------
        None
        """
        if isinstance(self.measure, str):
            try:
                measure = MEASURE_NAMES[self.measure]
            except KeyError:
                getLogger(__name__).error("No %r measure yet", self.measure)
                raise KeyError("No %r measure yet" % self.measure)

        getLogger(__name__).info(
            "Using MultivariateFilter with %s measure", measure)
        free_features = generate_features(X)
        self.selected_features_ = np.array([], dtype='int')

        relevance = np.apply_along_axis(
            mutual_information, 0, X[:, free_features], y)
        getLogger(__name__).info("Relevance vector: %s", relevance)

        redundancy = np.vectorize(
            lambda free_feature: matrix_mutual_information(
                X[:, free_features], X[:, free_feature]),
            signature='()->(1)')(free_features)
        getLogger(__name__).info("Redundancy vector: %s", redundancy)

        while len(self.selected_features_) != self.n_features:
            if self.beta is None:
                values = measure(
                    self.selected_features_, free_features, X, y,
                    relevance=relevance[free_features],
                    redundancy=np.sum(
                        redundancy[self.selected_features_],
                        axis=0)[free_features])
            else:
                if self.gamma is not None:
                    values = measure(
                        self.selected_features_, free_features, X, y, self.beta,
                        self.gamma, relevance=relevance[free_features],
                        redundancy=np.sum(
                            redundancy[self.selected_features_],
                            axis=0)[free_features])
                else:
                    values = measure(
                        self.selected_features_,free_features, X, y, self.beta,
                        relevance=relevance[free_features],
                        redundancy=np.sum(
                            redundancy[self.selected_features_],
                            axis=0)[free_features])

            getLogger(__name__).info("Free features: %s", free_features)
            getLogger(__name__).info("Measure values: %s", values)
            to_add = np.argmax(values)
            getLogger(__name__).info(
                "Adding feature %d to the selected set", free_features[to_add])
            self.selected_features_ = np.append(
                self.selected_features_, free_features[to_add])
            free_features = np.delete(free_features, to_add)


# %%
