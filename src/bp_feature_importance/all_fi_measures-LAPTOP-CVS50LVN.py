# %%
from cProfile import label
import time
from tqdm import tqdm
from rpy2.robjects.vectors import StrVector
import rpy2.robjects.packages as rpackages
import sys
import numpy as np
import pandas as pd
import shap
import random
import pickle
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from scipy.stats import entropy
import rpy2
from rpy2 import robjects
from rpy2.robjects.packages import importr
from sklearn import svm
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge, LinearRegression, Lasso, SGDClassifier, SGDRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from treeinterpreter import treeinterpreter
from sklearn.inspection import permutation_importance
from inspect import signature
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor, XGBRFClassifier, XGBRFRegressor
from fisher_score import fisherscore
from gplearn.genetic import SymbolicRegressor
from rfi import rfi, cfi
from fvecs.featurevec import FeatureVec
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, ExtraTreeRegressor, ExtraTreeClassifier
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

sys.path.append('E:/OneDrive/PhD/GitHub/Official_Dependency_Function/src/bp_dependency')
from dependency import convert_variable_to_prob_density_function

# %%
# Download R packages (copied from https://rpy2.github.io/doc/latest/html/introduction.html)
# import rpy2's package module
# import R's utility package
utils = rpackages.importr('utils')
# select a mirror for R packages
utils.chooseCRANmirror(ind=1)  # select the first mirror in the list
# R package names
packnames = ['infotheo', 'rfUtilities', 'randomForest', 'plyr']
packnames += ['HDclassif', 'KRLS', 'LiblineaR', 'LogicReg', 'RRF', 'RSNNS',
       'RWeka', 'Rborist', 'ada', 'adabag', 'bartMachine', 'binda',
       'bnclassify', 'brnn', 'bst', 'caTools', 'deepboost', 'deepnet',
       'evtree', 'extraTrees', 'fastAdaboost', 'fastICA', 'frbs', 'gam',
       'glmnet', 'h2o', 'hda', 'keras', 'kerndwd', 'kknn',
       'kohonen', 'leaps', 'monmlp', 'monomvn', 'msaenet', 'naivebayes',
       'neuralnet', 'nodeHarvest', 'obliqueRF', 'ordinalForest',
       'ordinalNet', 'pamr', 'penalized', 'penalizedLDA', 'plsRglm',
       'protoclass', 'qrnn', 'quantregForest', 'rFerns', 'randomGLM', 'relaxo', 'robustDA', 'rocc',
       'rotationForest', 'rpartScore', 'rqPen', 'rrcov', 'rrcovHD', 'sda',
       'sdwd', 'snn', 'sparseLDA', 'sparsediscrim', 'spikeslab', 'spls',
       'stepPlr', 'superpc', 'supervisedPRIM', 'wsrf', 'xgboost']

# R vector of strings
# Selectively install what needs to be install.
# We are fancy, just because we can.
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))


# %%
infotheo = importr('infotheo')
rfUtilities = importr('rfUtilities')
randomForest = importr('randomForest')
plyr = importr('plyr')
caret = importr('caret')
vip = importr('vip')
partykit = importr('partykit')
vita = importr('vita')
randomForest = importr('randomForest')
mixOmics = importr('mixOmics')
# for pack in packnames:
#     help_pack = importr(pack)
# %%

# TODO: this only works for 1d array
def convert_to_labelencoded(Y):
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(Y)
    label_encoded_y = label_encoder.transform(Y)
    return label_encoded_y

# TODO: this only works for 1d array


def convert_to_onehotencoded(Y):
    onehot_encoder = OneHotEncoder()
    onehot_encoder = onehot_encoder.fit(Y.reshape(-1, 1))
    onehot_encoded_y = onehot_encoder.transform(Y.reshape(-1, 1)).toarray()
    return onehot_encoded_y


def create_dataset(creation, save_path, **kwargs):
    # This will create a dataset in a certain path

    if creation == 'random_test':
        X_1 = np.random.randint(10, size=kwargs['n_observations'])
        X_2 = np.random.randint(10, size=kwargs['n_observations'])
        Y = X_1 + 2 * X_2
        X = np.stack((X_1, X_2), axis=1)
        dataset = np.stack((X_1, X_2, Y), axis=1)

    if creation == 'decimal_system':
        X_1 = np.random.randint(10, size=kwargs['n_observations'])
        X_2 = np.random.randint(10, size=kwargs['n_observations'])
        X_3 = np.random.randint(10, size=kwargs['n_observations'])
        Y = X_1 + 10 * X_2 + 100 * X_3
        X = np.stack((X_1, X_2, X_3), axis=1)
        dataset = np.stack((X_1, X_2, X_3, Y), axis=1)

    labelencoded_Y = convert_to_labelencoded(Y)
    onehotencoded_Y = convert_to_onehotencoded(Y)

    with open(save_path, 'wb') as f:
        pickle.dump([X, Y, labelencoded_Y, onehotencoded_Y, dataset], f)
    return


# %%
# create_dataset('decimal_system', 'datasets/decimal_system.pickle', n_observations=200)
# create_dataset('random_test', 'datasets/random_test.pickle', n_observations = 20000)

# %%


def load_dataset(data_path):
    # This will load the dataset in a certain path

    with open(data_path, 'rb') as f:
        X, Y, labelencoded_Y, onehotencoded_Y, dataset = pickle.load(f)

    if X.ndim == 1:
        X = np.expand_dims(X, axis=1)

    return X, Y, labelencoded_Y, onehotencoded_Y, dataset


# %%

# %%

# %%


def kwargs_reduced_to_func_arg(func, kwargs):
    sig = signature(func)
    overlap_args = list(set(kwargs.keys()) & set(sig.parameters.keys()))
    func_kwargs = {i: kwargs[i] for i in overlap_args}
    return func_kwargs


def parse_inputs(fi_method_name, **kwargs):
    # This will ensure that every input is in the correct form

    if fi_method_name == 'permutation_importance':
        reduced_kwargs = kwargs_reduced_to_func_arg(func=LogisticRegression, kwargs=kwargs)
        # Default model is Logistic Regression
        kwargs['model'] = kwargs.get('model', LogisticRegression(reduced_kwargs))

        # TODO dit nog fixen
        X, Y, labelencoded_Y, onehotencoded_Y, dataset = load_dataset(data_path='datasets/decimal_system.pickle')
        # If not fitted model is provided, fit the model
        kwargs['fitted_model'] = kwargs.get('fitted_model', kwargs['model'].fit(X, Y))

    # if 'shap' in fi_method_name:
    #     # If not fitted model is provided, fit the model
    #     kwargs['fitted_model'] = kwargs.get('fitted_model', kwargs['model'].fit(X,Y))

    return fi_method_name, kwargs


# %%
list_of_all_methods = []
# List of FI methods:
# from scikit-learn
# AdaBoost_Classifier (default)
list_of_all_methods += ['AdaBoost_Classifier']
# AdaBoost_Regressor (default)
list_of_all_methods += ['AdaBoost_Regressor']
# Random_Forest_Classifier (gini, entropy)
list_of_all_methods += ['Random_Forest_Classifier_' + i for i in ['gini', 'entropy']]
# Random_Forest_Regressor (gini, entropy)
list_of_all_methods += ['Random_Forest_Regressor']
# Extra_Trees_Classifier (gini, entropy)
list_of_all_methods += ['Extra_Trees_Classifier_' + i for i in ['gini', 'entropy']]
# Extra_Trees_Regressor (gini, entropy)
list_of_all_methods += ['Extra_Trees_Regressor']
# Gradient_Boosting_Classifier (default)
list_of_all_methods += ['Gradient_Boosting_Classifier']
# Gradient_Boosting_Regressor (default)
list_of_all_methods += ['Gradient_Boosting_Regressor']
# KL_divergence (default)
list_of_all_methods += ['KL_divergence']
# R_Mutual_Information (default)
list_of_all_methods += ['R_Mutual_Information']
# SVR_absolute_weights (rbf, linear)
list_of_all_methods += ['SVR_absolute_weights_' + i for i in ['linear']]
# EL_absolute_weights (default)
list_of_all_methods += ['EL_absolute_weights']
# Fisher_Score (default)
list_of_all_methods += ['Fisher_Score']
# permutation_importance_classifier
list_of_all_methods += ['permutation_importance_classifier_' + i for i in ['LogisticRegression', 'Ridge', 'LinearRegression',
                                                                'Lasso', 'SGDClassifier', 'SGDRegressor', 'MLPClassifier', 'MLPRegressor', 'SymbolicRegressor']]
# permutation_importance_regressor
list_of_all_methods += ['permutation_importance_regressor_' + i for i in ['LogisticRegression', 'Ridge', 'LinearRegression',
                                                                'Lasso', 'SGDClassifier', 'SGDRegressor', 'MLPClassifier', 'MLPRegressor', 'SymbolicRegressor']]
# shap_explainer_tree_classifier (XGBClassifier, XGBRFClassifier)
# TODO: Don't know if XGBRFClassifier works, as only zero FI are produced
list_of_all_methods += ['shap_explainer_tree_classifier_' + i for i in ['XGBClassifier', 'XGBRFClassifier']]
# shap_explainer_tree_regressor (XGBRegressor, XGBRFRegressor)
list_of_all_methods += ['shap_explainer_tree_regressor_' + i for i in ['XGBRegressor', 'XGBRFRegressor']]
# shap_explainer_linear_classifier (LogisticRegression, SGDClassifier)
list_of_all_methods += ['shap_explainer_linear_classifier_' + i for i in ['LogisticRegression', 'SGDClassifier']]
# shap_explainer_linear_regressor (Ridge, LinearRegression, Lasso, SGDRegressor)
list_of_all_methods += ['shap_explainer_linear_regressor_' + i for i in ['Ridge', 'LinearRegression', 'Lasso', 'SGDRegressor']]
# shap_explainer_permutation_classifier (LogisticRegression, SGDClassifier, MLPClassifier, XGBClassifier, XGBRFClassifier)
list_of_all_methods += ['shap_explainer_permutation_classifier_' +
                        i for i in ['LogisticRegression', 'SGDClassifier', 'MLPClassifier', 'XGBClassifier', 'XGBRFClassifier']]
# shap_explainer_permutation_regressor (Ridge, LinearRegression, Lasso, SGDRegressor, MLPRegressor, XGBRegressor, XGBRFRegressor, SymbolicRegressor)
list_of_all_methods += ['shap_explainer_permutation_regressor_' + i for i in ['Ridge', 'LinearRegression',
                                                                              'Lasso', 'SGDRegressor', 'MLPRegressor', 'XGBRegressor', 'XGBRFRegressor', 'SymbolicRegressor']]
# shap_explainer_partition_classifier (LogisticRegression, SGDClassifier, MLPClassifier, XGBClassifier, XGBRFClassifier)
list_of_all_methods += ['shap_explainer_partition_classifier_' + i for i in ['LogisticRegression',
                                                                             'SGDClassifier', 'MLPClassifier', 'XGBClassifier', 'XGBRFClassifier']]
# shap_explainer_partition_regressor (Ridge, LinearRegression, Lasso, SGDRegressor, MLPRegressor, XGBRegressor, XGBRFRegressor, SymbolicRegressor)
list_of_all_methods += ['shap_explainer_partition_regressor_' + i for i in ['Ridge', 'LinearRegression',
                                                                            'Lasso', 'SGDRegressor', 'MLPRegressor', 'XGBRegressor', 'XGBRFRegressor', 'SymbolicRegressor']]
# shap_explainer_sampling_classifier (LogisticRegression, SGDClassifier, MLPClassifier, XGBClassifier, XGBRFClassifier)
list_of_all_methods += ['shap_explainer_sampling_classifier_' + i for i in ['LogisticRegression',
                                                                            'SGDClassifier', 'MLPClassifier', 'XGBClassifier', 'XGBRFClassifier']]
# shap_explainer_sampling_regressor (Ridge, LinearRegression, Lasso, SGDRegressor, MLPRegressor, XGBRegressor, XGBRFRegressor, SymbolicRegressor)
list_of_all_methods += ['shap_explainer_sampling_regressor_' + i for i in ['Ridge', 'LinearRegression',
                                                                           'Lasso', 'SGDRegressor', 'MLPRegressor', 'XGBRegressor', 'XGBRFRegressor', 'SymbolicRegressor']]
# shap_explainer_kernel_classifier (LogisticRegression, SGDClassifier, MLPClassifier, XGBClassifier, XGBRFClassifier)
list_of_all_methods += ['shap_explainer_kernel_classifier_' + i for i in ['LogisticRegression',
                                                                          'SGDClassifier', 'MLPClassifier', 'XGBClassifier', 'XGBRFClassifier']]
# shap_explainer_kernel_regressor (Ridge, LinearRegression, Lasso, SGDRegressor, MLPRegressor, XGBRegressor, XGBRFRegressor, SymbolicRegressor)
list_of_all_methods += ['shap_explainer_kernel_regressor_' + i for i in ['Ridge', 'LinearRegression',
                                                                         'Lasso', 'SGDRegressor', 'MLPRegressor', 'XGBRegressor', 'XGBRFRegressor', 'SymbolicRegressor']]
# shap_explainer_exact_classifier (LogisticRegression, SGDClassifier, MLPClassifier, XGBClassifier, XGBRFClassifier)
list_of_all_methods += ['shap_explainer_exact_classifier_' + i for i in ['LogisticRegression',
                                                                         'SGDClassifier', 'MLPClassifier', 'XGBClassifier', 'XGBRFClassifier']]
# shap_explainer_exact_regressor (Ridge, LinearRegression, Lasso, SGDRegressor, MLPRegressor, XGBRegressor, XGBRFRegressor, SymbolicRegressor)
list_of_all_methods += ['shap_explainer_exact_regressor_' + i for i in ['Ridge', 'LinearRegression',
                                                                        'Lasso', 'SGDRegressor', 'MLPRegressor', 'XGBRegressor', 'XGBRFRegressor', 'SymbolicRegressor']]
#R_caret_classifier
list_of_all_methods += ['R_caret_classifier_' + i for i in
['snn', 'knn', 'bayesglm', 'lssvmRadial', 'rocc', 'ownn', 'ORFpls', 'rFerns', 'treebag', 'RRF', 'svmRadial', 'ctree2', 'evtree', 'pda', 'rpart', 'cforest', 'svmLinear', 'xyf', 'C5.0Tree', 'avNNet', 'kknn', 'svmRadialCost', 'gaussprRadial', 'FH.GBML', 'svmLinear2', 'bstSm', 'LogitBoost', 'wsrf', 'pls', 'plr', 'xgbLinear', 'rf', 'null', 'protoclass', 'monmlp', 'Rborist', 'mlpWeightDecay', 'svmRadialWeights', 'mlpML', 'ctree', 'loclda', 'sdwd', 'mlpWeightDecayML', 'svmRadialSigma', 'bstTree', 'dnn', 'ordinalRF', 'pda2', 'BstLm', 'RRFglobal', 'mlp', 'rpart1SE', 'pcaNNet', 'ORFsvm', 'parRF', 'rpart2', 'gaussprPoly', 'C5.0Rules', 'rda', 'rbfDDA', 'multinom', 'gaussprLinear', 'svmPoly']]
#R_caret_regressor
list_of_all_methods += ['R_caret_regressor_' + i for i in
['widekernelpls', 'pcr', 'knn', 'bayesglm', 'GFS.FR.MOGUL', 'qrnn', 'treebag', 'rqlasso', 'nnet', 'svmRadial', 'nnls', 'ctree2', 'evtree', 'rpart', 'cforest', 'svmLinear', 'enet', 'FIR.DM', 'xyf', 'HYFIS', 'leapSeq', 'glm', 'glm.nb', 'avNNet', 'kknn', 'svmRadialCost', 'gaussprRadial', 'ppr', 'DENFIS', 'svmLinear2', 'bstSm', 'lm', 'lars2', 'pls', 'rvmRadial', 'xgbLinear', 'simpls', 'rf', 'null', 'monmlp', 'Rborist', 'relaxo', 'GFS.THRIFT', 'mlpWeightDecay', 'randomGLM', 'mlpML', 'ctree', 'brnn', 'mlpWeightDecayML', 'kernelpls', 'krlsRadial', 'spikeslab', 'svmRadialSigma', 'lasso', 'glmnet', 'bstTree', 'dnn', 'icr', 'leapBackward', 'qrf', 'leapForward', 'BstLm', 'ANFIS', 'glmboost', 'mlp', 'rpart1SE', 'lmStepAIC', 'pcaNNet', 'lars', 'glmStepAIC', 'rpart2', 'gaussprPoly', 'ridge', 'FS.HGD', 'rbfDDA', 'gaussprLinear', 'svmPoly', 'penalized']]
#rfi_classifier
list_of_all_methods += ['rfi_classifier_' + i for i in ['LogisticRegression',
                                                                         'SGDClassifier', 'MLPClassifier', 'XGBClassifier', 'XGBRFClassifier']]
# rfi_regressor
list_of_all_methods += ['rfi_regressor_' + i for i in ['Ridge', 'LinearRegression',
                                                                        'Lasso', 'SGDRegressor', 'MLPRegressor', 'XGBRegressor', 'XGBRFRegressor', 'SymbolicRegressor']]
#cfi_classifier
list_of_all_methods += ['cfi_classifier_' + i for i in ['LogisticRegression',
                                                                         'SGDClassifier', 'MLPClassifier', 'XGBClassifier', 'XGBRFClassifier']]
# cfi_regressor
list_of_all_methods += ['cfi_regressor_' + i for i in ['Ridge', 'LinearRegression',
                                                                        'Lasso', 'SGDRegressor', 'MLPRegressor', 'XGBRegressor', 'XGBRFRegressor', 'SymbolicRegressor']]
# featurevec_classifier
list_of_all_methods += ['featurevec_classifier']
 # featurevec_regressor
list_of_all_methods += ['featurevec_regressor']
#R_firm_classifier
list_of_all_methods += ['R_firm_classifier_' + i for i in
['knn', 'treebag', 'RRF', 'ctree2', 'evtree', 'pda', 'rpart', 'cforest', 'xyf', 'C5.0Tree', 'avNNet', 'kknn', 'gaussprRadial', 'LogitBoost', 'wsrf', 'pls', 'xgbLinear', 'rf', 'null', 'monmlp', 'Rborist', 'mlpWeightDecay', 'mlpML', 'ctree', 'mlpWeightDecayML', 'dnn', 'pda2', 'RRFglobal', 'mlp', 'rpart1SE', 'parRF', 'rpart2', 'gaussprPoly', 'C5.0Rules', 'rbfDDA', 'multinom', 'gaussprLinear']]
#R_firm_regressor
list_of_all_methods += ['R_firm_regressor_' + i for i in
['widekernelpls', 'pcr', 'knn', 'bayesglm', 'GFS.FR.MOGUL', 'qrnn', 'treebag', 'rqlasso', 'nnet', 'svmRadial', 'nnls', 'ctree2', 'evtree', 'rpart', 'cforest', 'svmLinear', 'enet', 'FIR.DM', 'xyf', 'HYFIS', 'leapSeq', 'glm', 'glm.nb', 'avNNet', 'kknn', 'svmRadialCost', 'gaussprRadial', 'ppr', 'DENFIS', 'svmLinear2', 'bstSm', 'lm', 'lars2', 'pls', 'rvmRadial', 'xgbLinear', 'simpls', 'rf', 'null', 'monmlp', 'Rborist', 'relaxo', 'GFS.THRIFT', 'mlpWeightDecay', 'randomGLM', 'mlpML', 'ctree', 'brnn', 'mlpWeightDecayML', 'kernelpls', 'krlsRadial', 'spikeslab', 'svmRadialSigma', 'lasso', 'glmnet', 'bstTree', 'dnn', 'icr', 'leapBackward', 'qrf', 'leapForward', 'BstLm', 'ANFIS', 'glmboost', 'mlp', 'rpart1SE', 'lmStepAIC', 'pcaNNet', 'lars', 'glmStepAIC', 'rpart2', 'gaussprPoly', 'ridge', 'FS.HGD', 'rbfDDA', 'gaussprLinear', 'svmPoly', 'penalized']]
# PCA
list_of_all_methods += ['PCA_sum']
# PCA_weighted
list_of_all_methods += ['PCA_weighted']
# varimp_classifier
list_of_all_methods += ['R_varimp_classifier']
# varimp_regressor
list_of_all_methods += ['R_varimp_regressor']
# pimp_classifier
list_of_all_methods += ['R_pimp_classifier']
# pimp_regressor
list_of_all_methods += ['R_pimp_regressor']
# vip_sum_classifier
list_of_all_methods += ['R_vip_sum_classifier_' + i for i in ['plsda', 'splsda']]
# vip_sum_regressor
list_of_all_methods += ['R_vip_sum_regressor_' + i for i in ['pls', 'spls']]
# vip_weighted_X_classifier
list_of_all_methods += ['R_vip_weighted_X_classifier_' + i for i in ['plsda', 'splsda']]
# vip_weighted_X_regressor
list_of_all_methods += ['R_vip_weighted_X_regressor_' + i for i in ['pls', 'spls']]
# vip_weighted_Y_classifier
list_of_all_methods += ['R_vip_weighted_Y_classifier_' + i for i in ['plsda', 'splsda']]
# vip_weighted_Y_regressor
list_of_all_methods += ['R_vip_weighted_Y_regressor_' + i for i in ['pls', 'spls']]
# treeinterpreter_classifier
list_of_all_methods += ['treeinterpreter_classifier_' + i for i in ['DecisionTreeClassifier',
'ExtraTreeClassifier', 'RandomForestClassifier', 'ExtraTreesClassifier', 'ExtraTreeClassifier']]
# treeinterpreter_regressor
list_of_all_methods += ['treeinterpreter_regressor_' + i for i in ['DecisionTreeRegressor', 'ExtraTreeRegressor', 'RandomForestRegressor', 'ExtraTreesRegressor', 'ExtraTreeRegressor']]
# %%


def initialize_experiment_variables(name, X, Y, labelencoded_Y):
    kwargs = {}

    ###########################################################
    # We only use the default values for the following methods#
    ###########################################################

    if 'AdaBoost_Classifier' in name:
        fi_method_name = 'AdaBoost_Classifier'
        # default values:
        kwargs['base_estimator'] = None
        kwargs['n_estimators'] = 50
        kwargs['learning_rate'] = 1.0
        kwargs['algorithm'] = 'SAMME.R'
        kwargs['random_state'] = None

    if 'AdaBoost_Regressor' in name:
        fi_method_name = 'AdaBoost_Regressor'
        # default values:
        kwargs['base_estimator'] = None
        kwargs['n_estimators'] = 50
        kwargs['learning_rate'] = 1.0
        kwargs['loss'] = 'linear'
        kwargs['random_state'] = None

    if 'Random_Forest_Regressor' in name:
        fi_method_name = 'Random_Forest_Regressor'
        # default values:
        kwargs['n_estimators'] = 100
        kwargs['criterion'] = 'squared_error'
        kwargs['max_depth'] = None
        kwargs['min_samples_split'] = 2
        kwargs['min_samples_leaf'] = 1
        kwargs['min_weight_fraction_leaf'] = 0.0
        kwargs['max_features'] = 'auto'
        kwargs['max_leaf_nodes'] = None
        kwargs['min_impurity_decrease'] = 0.0
        kwargs['bootstrap'] = True
        kwargs['oob_score'] = False
        kwargs['n_jobs'] = None
        kwargs['random_state'] = None
        kwargs['verbose'] = 0
        kwargs['warm_start'] = False
        kwargs['ccp_alpha'] = 0.0
        kwargs['max_samples'] = None

    if 'Extra_Trees_Regressor' in name:
        fi_method_name = 'Extra_Trees_Regressor'
        # default values:
        kwargs['n_estimators'] = 100
        kwargs['criterion'] = 'squared_error'
        kwargs['max_depth'] = None
        kwargs['min_samples_split'] = 2
        kwargs['min_samples_leaf'] = 1
        kwargs['min_weight_fraction_leaf'] = 0.0
        kwargs['max_features'] = 'auto'
        kwargs['max_leaf_nodes'] = None
        kwargs['min_impurity_decrease'] = 0.0
        kwargs['bootstrap'] = False
        kwargs['oob_score'] = False
        kwargs['n_jobs'] = None
        kwargs['random_state'] = None
        kwargs['verbose'] = 0
        kwargs['warm_start'] = False
        kwargs['ccp_alpha'] = 0.0
        kwargs['max_samples'] = None

    if 'Gradient_Boosting_Classifier' in name:
        fi_method_name = 'Gradient_Boosting_Classifier'
        # default values:
        kwargs['loss'] = 'deviance'
        kwargs['learning_rate'] = 0.1
        kwargs['n_estimators'] = 100
        kwargs['subsample'] = 1.0
        kwargs['criterion'] = 'friedman_mse'
        kwargs['min_samples_split'] = 2
        kwargs['min_samples_leaf'] = 1
        kwargs['min_weight_fraction_leaf'] = 0.0
        kwargs['max_depth'] = 3
        kwargs['min_impurity_decrease'] = 0.0
        kwargs['init'] = None
        kwargs['random_state'] = None
        kwargs['max_features'] = None
        kwargs['verbose'] = 0
        kwargs['max_leaf_nodes'] = None
        kwargs['warm_start'] = False
        kwargs['validation_fraction'] = 0.1
        kwargs['n_iter_no_change'] = None
        kwargs['tol'] = 0.0001
        kwargs['ccp_alpha'] = 0.0

    if 'Gradient_Boosting_Regressor' in name:
        fi_method_name = 'Gradient_Boosting_Regressor'
        # default values:
        kwargs['loss'] = 'squared_error'
        kwargs['learning_rate'] = 0.1
        kwargs['n_estimators'] = 100
        kwargs['subsample'] = 1.0
        kwargs['criterion'] = 'friedman_mse'
        kwargs['min_samples_split'] = 2
        kwargs['min_samples_leaf'] = 1
        kwargs['min_weight_fraction_leaf'] = 0.0
        kwargs['max_depth'] = 3
        kwargs['min_impurity_decrease'] = 0.0
        kwargs['init'] = None
        kwargs['random_state'] = None
        kwargs['max_features'] = None
        kwargs['alpha'] = 0.9
        kwargs['verbose'] = 0
        kwargs['max_leaf_nodes'] = None
        kwargs['warm_start'] = False
        kwargs['validation_fraction'] = 0.1
        kwargs['n_iter_no_change'] = None
        kwargs['tol'] = 0.0001
        kwargs['ccp_alpha'] = 0.0

    if 'KL_divergence' in name:
        fi_method_name = 'KL_divergence'
        # default values:
        kwargs['base'] = None
        kwargs['axis'] = 0

    if 'R_Mutual_Information' in name:
        fi_method_name = 'R_Mutual_Information'
        # default values:
        kwargs['method'] = 'emp'

    if 'EL_absolute_weights' in name:
        fi_method_name = 'EL_absolute_weights'
        # default values:
        kwargs['alpha'] = 1.0
        kwargs['l1_ratio'] = 0.5
        kwargs['fit_intercept'] = True
        kwargs['normalize'] = 'deprecated'
        kwargs['precompute'] = False
        kwargs['max_iter'] = 1000
        kwargs['copy_X'] = True
        kwargs['tol'] = 0.0001
        kwargs['warm_start'] = False
        kwargs['positive'] = False
        kwargs['random_state'] = None
        kwargs['selection'] = 'cyclic'

    if 'Fisher_Score' in name:
        fi_method_name = 'Fisher_Score'
        # default values:


    if 'featurevec_classifier' in name:
        fi_method_name = 'featurevec_classifier'

    if 'featurevec_regressor' in name:
        fi_method_name = 'featurevec_regressor'

    if 'PCA_sum' == name:
        fi_method_name = 'PCA_sum'

    if 'PCA_weighted' == name:
        fi_method_name = 'PCA_weighted'

    if 'R_varimp_classifier' == name:
        fi_method_name = 'R_varimp_classifier'

    if 'R_varimp_regressor' == name:
        fi_method_name = 'R_varimp_regressor'

    if 'R_pimp_classifier' == name:
        fi_method_name = 'R_pimp_classifier'

    if 'R_pimp_regressor' == name:
        fi_method_name = 'R_pimp_regressor'


    ###########################################################
    # We use different parameters for the following methods#
    ###########################################################

    if 'Random_Forest_Classifier' in name:
        fi_method_name = 'Random_Forest_Classifier'
        # default values:
        kwargs['n_estimators'] = 100
        kwargs['criterion'] = 'gini'
        kwargs['max_depth'] = None
        kwargs['min_samples_split'] = 2
        kwargs['min_samples_leaf'] = 1
        kwargs['min_weight_fraction_leaf'] = 0.0
        kwargs['max_features'] = 'auto'
        kwargs['max_leaf_nodes'] = None
        kwargs['min_impurity_decrease'] = 0.0
        kwargs['bootstrap'] = True
        kwargs['oob_score'] = False
        kwargs['n_jobs'] = None
        kwargs['random_state'] = None
        kwargs['verbose'] = 0
        kwargs['warm_start'] = False
        kwargs['class_weight'] = None
        kwargs['ccp_alpha'] = 0.0
        kwargs['max_samples'] = None

        # custom values:
        if 'gini' in name:
            kwargs['criterion'] = 'gini'  # is also default

        if 'entropy' in name:
            kwargs['criterion'] = 'entropy'

    if 'Extra_Trees_Classifier' in name:
        fi_method_name = 'Extra_Trees_Classifier'
        # default values:
        kwargs['n_estimators'] = 100
        kwargs['criterion'] = 'gini'
        kwargs['max_depth'] = None
        kwargs['min_samples_split'] = 2
        kwargs['min_samples_leaf'] = 1
        kwargs['min_weight_fraction_leaf'] = 0.0
        kwargs['max_features'] = 'auto'
        kwargs['max_leaf_nodes'] = None
        kwargs['min_impurity_decrease'] = 0.0
        kwargs['bootstrap'] = False
        kwargs['oob_score'] = False
        kwargs['n_jobs'] = None
        kwargs['random_state'] = None
        kwargs['verbose'] = 0
        kwargs['warm_start'] = False
        kwargs['class_weight'] = None
        kwargs['ccp_alpha'] = 0.0
        kwargs['max_samples'] = None

        # custom values:
        if 'gini' in name:
            kwargs['criterion'] = 'gini'  # is also default

        if 'entropy' in name:
            kwargs['criterion'] = 'entropy'

    if 'SVR_absolute_weights' in name:
        fi_method_name = 'SVR_absolute_weights'
        # default values:
        kwargs['kernel'] = 'rbf'
        kwargs['degree'] = 3
        kwargs['gamma'] = 'scale'
        kwargs['coef0'] = 0.0
        kwargs['tol'] = 0.001
        kwargs['C'] = 1.0
        kwargs['epsilon'] = 0.1
        kwargs['shrinking'] = True
        kwargs['cache_size'] = 200
        kwargs['verbose'] = False
        kwargs['max_iter'] = - 1

        # custom values:
        # coef are only available for a linear kernel
        if 'linear' in name:
            kwargs['kernel'] = 'linear'


    if 'permutation_importance_classifier' in name:
        fi_method_name = 'permutation_importance_classifier'

        # default values:
        kwargs['scoring'] = None
        kwargs['n_repeats'] = 5
        kwargs['n_jobs'] = None
        kwargs['random_state'] = None
        kwargs['sample_weight'] = None
        kwargs['max_samples'] = 1.0

        # custom values:
        kwargs['model'] = eval(name.split(fi_method_name + '_', 1)[1])()

    if 'permutation_importance_regressor' in name:
        fi_method_name = 'permutation_importance_regressor'

        # default values:
        kwargs['scoring'] = None
        kwargs['n_repeats'] = 5
        kwargs['n_jobs'] = None
        kwargs['random_state'] = None
        kwargs['sample_weight'] = None
        kwargs['max_samples'] = 1.0

        # custom values:
        kwargs['model'] = eval(name.split(fi_method_name + '_', 1)[1])()

    if 'shap_explainer_tree_classifier' in name:
        fi_method_name = 'shap_explainer_tree_classifier'
        # default values:

        # custom values:
        kwargs['model'] = eval(name.split(fi_method_name + '_', 1)[1])()


    if 'shap_explainer_tree_regressor' in name:
        fi_method_name = 'shap_explainer_tree_regressor'
        # default values:

        # custom values:
        kwargs['model'] = eval(name.split(fi_method_name + '_', 1)[1])()

    if 'shap_explainer_linear_classifier' in name:
        fi_method_name = 'shap_explainer_linear_classifier'
        # default values:

        # custom values:
        kwargs['model'] = eval(name.split(fi_method_name + '_', 1)[1])()

    if 'shap_explainer_linear_regressor' in name:
        fi_method_name = 'shap_explainer_linear_regressor'
        # default values:

        # custom values:
        kwargs['model'] = eval(name.split(fi_method_name + '_', 1)[1])()

    if 'shap_explainer_permutation_classifier' in name:
        fi_method_name = 'shap_explainer_permutation_classifier'
        # default values:

        # custom values:
        kwargs['model'] = eval(name.split(fi_method_name + '_', 1)[1])()

    if 'shap_explainer_permutation_regressor' in name:
        fi_method_name = 'shap_explainer_permutation_regressor'
        # default values:

        # custom values:
        kwargs['model'] = eval(name.split(fi_method_name + '_', 1)[1])()

    if 'shap_explainer_partition_classifier' in name:
        fi_method_name = 'shap_explainer_partition_classifier'
        # default values:

        # custom values:
        kwargs['model'] = eval(name.split(fi_method_name + '_', 1)[1])()

    if 'shap_explainer_partition_regressor' in name:
        fi_method_name = 'shap_explainer_partition_regressor'
        # default values:

        # custom values:
        kwargs['model'] = eval(name.split(fi_method_name + '_', 1)[1])()

    if 'shap_explainer_sampling_classifier' in name:
        fi_method_name = 'shap_explainer_sampling_classifier'
        # default values:

        # custom values:
        kwargs['model'] = eval(name.split(fi_method_name + '_', 1)[1])()

    if 'shap_explainer_sampling_regressor' in name:
        fi_method_name = 'shap_explainer_sampling_regressor'
        # default values:

        # custom values:
        kwargs['model'] = eval(name.split(fi_method_name + '_', 1)[1])()

    if 'shap_explainer_kernel_classifier' in name:
        fi_method_name = 'shap_explainer_kernel_classifier'
        # default values:

        # custom values:
        kwargs['model'] = eval(name.split(fi_method_name + '_', 1)[1])()

    if 'shap_explainer_kernel_regressor' in name:
        fi_method_name = 'shap_explainer_kernel_regressor'
        # default values:

        # custom values:
        kwargs['model'] = eval(name.split(fi_method_name + '_', 1)[1])()

    if 'shap_explainer_exact_classifier' in name:
        fi_method_name = 'shap_explainer_exact_classifier'
        # default values:

        # custom values:
        kwargs['model'] = eval(name.split(fi_method_name + '_', 1)[1])()

    if 'shap_explainer_exact_regressor' in name:
        fi_method_name = 'shap_explainer_exact_regressor'
        # default values:

        # custom values:
        kwargs['model'] = eval(name.split(fi_method_name + '_', 1)[1])()

    if 'R_caret_classifier' in name:
        fi_method_name = 'R_caret_classifier'
        kwargs['method_name'] = name.split('R_caret_classifier_', 1)[1]

    if 'R_caret_regressor' in name:
        fi_method_name = 'R_caret_regressor'
        kwargs['method_name'] = name.split('R_caret_regressor_', 1)[1]

    if 'rfi_classifier' in name:
        fi_method_name = 'rfi_classifier'
        kwargs['loss'] = lambda x, y : np.power(x-y, 2)
        kwargs['G'] = np.array([])
        kwargs['D'] = np.arange(0, X.shape[1])
        kwargs['n_repeats'] = 30

        kwargs['model'] = eval(name.split(fi_method_name + '_', 1)[1])()

    if 'rfi_regressor' in name:
        fi_method_name = 'rfi_regressor'
        kwargs['loss'] = lambda x, y : np.power(x-y, 2)
        kwargs['G'] = np.array([])
        kwargs['D'] = np.arange(0, X.shape[1])
        kwargs['n_repeats'] = 30

        kwargs['model'] = eval(name.split(fi_method_name + '_', 1)[1])()

    if 'cfi_classifier' in name:
        fi_method_name = 'cfi_classifier'
        kwargs['loss'] = lambda x, y : np.power(x-y, 2)
        kwargs['D'] = np.arange(0, X.shape[1])
        kwargs['n_repeats'] = 30

        kwargs['model'] = eval(name.split(fi_method_name + '_', 1)[1])()


    if 'cfi_regressor' in name:
        fi_method_name = 'cfi_regressor'
        kwargs['loss'] = lambda x, y : np.power(x-y, 2)
        kwargs['D'] = np.arange(0, X.shape[1])
        kwargs['n_repeats'] = 30

        kwargs['model'] = eval(name.split(fi_method_name + '_', 1)[1])()



    if 'R_firm_classifier' in name:
        fi_method_name = 'R_firm_classifier'
        kwargs['method_name'] = name.split('R_firm_classifier_', 1)[1]

    if 'R_firm_regressor' in name:
        fi_method_name = 'R_firm_regressor'
        kwargs['method_name'] = name.split('R_firm_regressor_', 1)[1]

    if 'R_vip_sum_classifier' in name:
        fi_method_name = 'R_vip_sum_classifier'
        kwargs['method_name'] = name.split('R_vip_sum_classifier_', 1)[1]

    if 'R_vip_sum_regressor' in name:
        fi_method_name = 'R_vip_sum_regressor'
        kwargs['method_name'] = name.split('R_vip_sum_regressor_', 1)[1]

    if 'R_vip_weighted_X_classifier' in name:
        fi_method_name = 'R_vip_weighted_X_classifier'
        kwargs['method_name'] = name.split('R_vip_weighted_X_classifier_', 1)[1]

    if 'R_vip_weighted_X_regressor' in name:
        fi_method_name = 'R_vip_weighted_X_regressor'
        kwargs['method_name'] = name.split('R_vip_weighted_X_regressor_', 1)[1]

    if 'R_vip_weighted_Y_classifier' in name:
        fi_method_name = 'R_vip_weighted_Y_classifier'
        kwargs['method_name'] = name.split('R_vip_weighted_Y_classifier_', 1)[1]

    if 'R_vip_weighted_Y_regressor' in name:
        fi_method_name = 'R_vip_weighted_Y_regressor'
        kwargs['method_name'] = name.split('R_vip_weighted_Y_regressor_', 1)[1]

    if 'treeinterpreter_classifier' in name:
        fi_method_name = 'treeinterpreter_classifier'
        kwargs['model'] = eval(name.split(fi_method_name + '_', 1)[1])()

    if 'treeinterpreter_regressor' in name:
        fi_method_name = 'treeinterpreter_regressor'
        kwargs['model'] = eval(name.split(fi_method_name + '_', 1)[1])()


    return kwargs, fi_method_name

# %%


def convert_average_by_class(Y, X, model, average_abs_v):
    # This is conditioned for each class. We average using the probability that each class is observed prob Y * (FI | Y) doen.
    Y_prob = convert_variable_to_prob_density_function(Y)
    fi_results = np.zeros(X.shape[1])
    for i, j in enumerate(model.classes_):
        if Y.ndim == 1:
            fi_results += Y_prob[(j,)] * average_abs_v[..., i]
        else:
            # TODO: checken of dit klopt, of dat het j, moet zijn net als hierboven
            fi_results += Y_prob[tuple(j)] * average_abs_v[..., i]

    return fi_results


def determine_fi(fi_method_name, data_path, **kwargs):

    X, Y, labelencoded_Y, onehotencoded_Y, dataset = load_dataset(data_path=data_path)
    if len(kwargs) > 0:
        fi_method_name, kwargs = parse_inputs(fi_method_name, **kwargs)

    # AdaBoost_Classifier
    if fi_method_name == 'AdaBoost_Classifier':
        clf = AdaBoostClassifier(**kwargs_reduced_to_func_arg(func=AdaBoostClassifier, kwargs=kwargs))
        clf.fit(X, labelencoded_Y)
        fi_results = clf.feature_importances_

    # AdaBoost_Regressor
    if fi_method_name == 'AdaBoost_Regressor':
        regr = AdaBoostRegressor(**kwargs_reduced_to_func_arg(func=AdaBoostRegressor, kwargs=kwargs))
        regr.fit(X, Y)
        fi_results = regr.feature_importances_

    # Random_Forest_Classifier
    if fi_method_name == 'Random_Forest_Classifier':
        clf = RandomForestClassifier(**kwargs_reduced_to_func_arg(func=RandomForestClassifier, kwargs=kwargs))
        clf.fit(X, labelencoded_Y)
        fi_results = clf.feature_importances_

    # Random_Forest_Regressor
    if fi_method_name == 'Random_Forest_Regressor':
        regr = RandomForestRegressor(**kwargs_reduced_to_func_arg(func=RandomForestRegressor, kwargs=kwargs))
        regr.fit(X, Y)
        fi_results = regr.feature_importances_

    # Extra_Trees_Classifier
    if fi_method_name == 'Extra_Trees_Classifier':
        clf = ExtraTreesClassifier(**kwargs_reduced_to_func_arg(func=ExtraTreesClassifier, kwargs=kwargs))
        clf.fit(X, labelencoded_Y)
        fi_results = clf.feature_importances_

    # Extra_Trees_Regressor
    if fi_method_name == 'Extra_Trees_Regressor':
        regr = ExtraTreesRegressor(**kwargs_reduced_to_func_arg(func=ExtraTreesRegressor, kwargs=kwargs))
        regr.fit(X, Y)
        fi_results = regr.feature_importances_

    # Gradient_Boosting_Classifier
    if fi_method_name == 'Gradient_Boosting_Classifier':
        clf = GradientBoostingClassifier(**kwargs_reduced_to_func_arg(func=GradientBoostingClassifier, kwargs=kwargs))
        clf.fit(X, labelencoded_Y)
        fi_results = clf.feature_importances_

    # Gradient_Boosting_Regressor
    if fi_method_name == 'Gradient_Boosting_Regressor':
        regr = GradientBoostingRegressor(**kwargs_reduced_to_func_arg(func=GradientBoostingRegressor, kwargs=kwargs))
        regr.fit(X, Y)
        fi_results = regr.feature_importances_

    # KL_divergence
    if fi_method_name == 'KL_divergence':
        # If Y does not have the same dimension as X[:, i], problems can occur -> np.NaN
        fi_results = [None] * X.shape[1]
        for i in range(X.shape[1]):
            try:
                fi_results[i] = entropy(pk=X[:, i], qk=Y, **kwargs_reduced_to_func_arg(func=entropy, kwargs=kwargs))
            except:
                fi_results[i] = np.NaN
        fi_results = np.array(fi_results)

    # R_Mutual_Information
    # Method is implemented in R
    if fi_method_name == 'R_Mutual_Information':
        mutinformation = robjects.r['mutinformation']
        # If Y does not have the same dimension as X[:, i], problems can occur -> np.NaN
        R_Y = robjects.FloatVector(Y)
        fi_results = [None] * X.shape[1]
        for i in range(X.shape[1]):
            R_X = robjects.FloatVector(X[:, i])
            try:
                fi_results[i] = mutinformation(R_X, R_Y, **kwargs_reduced_to_func_arg(func=mutinformation, kwargs=kwargs))
            except:
                fi_results[i] = np.NaN
        fi_results = np.array(fi_results).squeeze()

    # SVR_absolute_weights
    if fi_method_name == 'SVR_absolute_weights':
        regr = svm.SVR(**kwargs_reduced_to_func_arg(func=svm.SVR, kwargs=kwargs))
        regr.fit(X, Y)
        fi_results = np.absolute(regr.coef_).squeeze()

    # EL_absolute_weights
    if fi_method_name == 'EL_absolute_weights':
        regr = ElasticNet(**kwargs_reduced_to_func_arg(func=ElasticNet, kwargs=kwargs))
        regr.fit(X, Y)
        fi_results = np.absolute(regr.coef_)

    # Fisher_Score
    # TODO: ik krijg nog divided by zero errors. Is dat een fout of klopt dat?
    if fi_method_name == 'Fisher_Score':
        FS = fisherscore(X, Y)
        fi_results = np.array(FS.fisher_score_list)

    # permutation_importance_classifier
    if fi_method_name == 'permutation_importance_classifier':
        reduced_kwargs = kwargs_reduced_to_func_arg(func=permutation_importance, kwargs=kwargs)
        fitted_model = kwargs['model'].fit(X, labelencoded_Y)
        fi_results = permutation_importance(estimator= fitted_model, X=X, y= labelencoded_Y, **reduced_kwargs).importances_mean

    if fi_method_name == 'permutation_importance_regressor':
        reduced_kwargs = kwargs_reduced_to_func_arg(func=permutation_importance, kwargs=kwargs)
        fitted_model = kwargs['model'].fit(X, Y)
        fi_results = permutation_importance(estimator= fitted_model, X=X, y= Y, **reduced_kwargs).importances_mean

    if fi_method_name == 'shap_explainer_tree_classifier':
        model = kwargs['model'].fit(X, labelencoded_Y)
        explainer = shap.explainers.Tree(model, X)
        shap_values = explainer(X)
        average_abs_shap_v = shap_values.abs.values.mean(0)
        fi_results = convert_average_by_class(labelencoded_Y, X, model, average_abs_shap_v)

    if fi_method_name == 'shap_explainer_tree_regressor':
        model = kwargs['model'].fit(X, Y)
        explainer = shap.explainers.Tree(model, X)
        shap_values = explainer(X)
        fi_results = shap_values.abs.values.mean(0)

    if fi_method_name == 'shap_explainer_linear_classifier':
        model = kwargs['model'].fit(X, labelencoded_Y)
        explainer = shap.explainers.Linear(model, X)
        shap_values = explainer(X)
        average_abs_shap_v = shap_values.abs.values.mean(0)
        fi_results = convert_average_by_class(labelencoded_Y, X, model, average_abs_shap_v)

    if fi_method_name == 'shap_explainer_linear_regressor':
        model = kwargs['model'].fit(X, Y)
        explainer = shap.explainers.Linear(model, X)
        shap_values = explainer(X)
        fi_results = shap_values.abs.values.mean(0)

    if fi_method_name == 'shap_explainer_permutation_classifier':
        model = kwargs['model'].fit(X, labelencoded_Y)
        explainer = shap.explainers.Permutation(model.predict, X)
        shap_values = explainer(X)
        fi_results = shap_values.abs.values.mean(0)

    if fi_method_name == 'shap_explainer_permutation_regressor':
        model = kwargs['model'].fit(X, Y)
        explainer = shap.explainers.Permutation(model.predict, X)
        shap_values = explainer(X)
        fi_results = shap_values.abs.values.mean(0)

    if fi_method_name == 'shap_explainer_partition_classifier':
        model = kwargs['model'].fit(X, labelencoded_Y)
        explainer = shap.explainers.Partition(model.predict, X)
        shap_values = explainer(X)
        fi_results = shap_values.abs.values.mean(0)

    if fi_method_name == 'shap_explainer_partition_regressor':
        model = kwargs['model'].fit(X, Y)
        explainer = shap.explainers.Partition(model.predict, X)
        shap_values = explainer(X)
        fi_results = shap_values.abs.values.mean(0)

    if fi_method_name == 'shap_explainer_sampling_classifier':
        model = kwargs['model'].fit(X, labelencoded_Y)
        explainer = shap.explainers.Sampling(model.predict, X)
        shap_values = explainer(X)
        fi_results = shap_values.abs.values.mean(0)

    if fi_method_name == 'shap_explainer_sampling_regressor':
        model = kwargs['model'].fit(X, Y)
        explainer = shap.explainers.Sampling(model.predict, X)
        shap_values = explainer(X)
        fi_results = shap_values.abs.values.mean(0)

    if fi_method_name == 'shap_explainer_kernel_classifier':
        model = kwargs['model'].fit(X, labelencoded_Y)
        explainer = shap.KernelExplainer(model.predict, X)
        shap_values = explainer.shap_values(X)
        fi_results = np.mean(np.abs(shap_values), axis=0)

    if fi_method_name == 'shap_explainer_kernel_regressor':
        model = kwargs['model'].fit(X, Y)
        explainer = shap.KernelExplainer(model.predict, X)
        shap_values = explainer.shap_values(X)
        fi_results = np.mean(np.abs(shap_values), axis=0)

    if fi_method_name == 'shap_explainer_exact_classifier':
        model = kwargs['model'].fit(X, labelencoded_Y)
        explainer = shap.explainers.Exact(model.predict, X)
        shap_values = explainer(X)
        fi_results = shap_values.abs.values.mean(0)

    if fi_method_name == 'shap_explainer_exact_regressor':
        model = kwargs['model'].fit(X, Y)
        explainer = shap.explainers.Exact(model.predict, X)
        shap_values = explainer(X)
        fi_results = shap_values.abs.values.mean(0)

    #TODO: Works only with 1D Y as it unlists
    if fi_method_name == 'R_caret_classifier':
        robjects.globalenv["X"] = X
        robjects.globalenv["Y"] = Y
        r_return = robjects.r('''
        X <- as.data.frame(X)
        Y <- as.numeric(array(unlist(Y)))
        as_factor_Y <- as.factor(Y)
        ''')
        robjects.globalenv["method_tested"] = kwargs['method_name']
        fi_results = robjects.r('''
        fitControl <- trainControl(method = "none", classProbs = F)
        fitted_model <- train(x= X, y= as_factor_Y,
                method = method_tested,
                trControl = fitControl
                )
        Imp <- varImp(fitted_model, scale = FALSE)
        b <-  Imp$importance
        a <- (count(as_factor_Y)$freq / sum(count(as_factor_Y)$freq))
        as.vector(rowSums(b * a))
        ''')
        fi_results = np.array(fi_results)

    #TODO: Works only with 1D Y as it unlists
    if fi_method_name == 'R_caret_regressor':
        robjects.globalenv["X"] = X
        robjects.globalenv["Y"] = Y
        r_return = robjects.r('''
        X <- as.data.frame(X)
        Y <- as.numeric(array(unlist(Y)))
        ''')
        robjects.globalenv["method_tested"] = kwargs['method_name']
        fi_results = robjects.r('''
        fitControl <- trainControl(method = "none", classProbs = F)
        fitted_model <- train(x= X, y= Y,
                method = method_tested,
                trControl = fitControl)
        Imp <- varImp(fitted_model, scale = FALSE)
        as.vector(Imp$importance$Overall)
        ''')
        fi_results = np.array(fi_results)


    if fi_method_name == 'rfi_classifier':
        model = kwargs['model'].fit(X, labelencoded_Y)
        fi_results = rfi(predict= model.predict, X_train= X, X_test= X, y_test= labelencoded_Y, **kwargs_reduced_to_func_arg(func=rfi, kwargs=kwargs))[0]

    if fi_method_name == 'rfi_regressor':
        model = kwargs['model'].fit(X, Y)
        fi_results = rfi(predict= model.predict, X_train= X, X_test= X, y_test= Y, **kwargs_reduced_to_func_arg(func=rfi, kwargs=kwargs))[0]

    if fi_method_name == 'cfi_classifier':
        model = kwargs['model'].fit(X, labelencoded_Y)
        fi_results = cfi(predict= model.predict, X_train= X, X_test= X, y_test= labelencoded_Y, **kwargs_reduced_to_func_arg(func=cfi, kwargs=kwargs))[0]

    if fi_method_name == 'cfi_regressor':
        model = kwargs['model'].fit(X, Y)
        fi_results = cfi(predict= model.predict, X_train= X, X_test= X, y_test= Y, **kwargs_reduced_to_func_arg(func=cfi, kwargs=kwargs))[0]

    if fi_method_name == 'featurevec_classifier':
        fv = FeatureVec(mode='classify', feature_names= range(0, X.shape[1]))
        fv.fit(X, labelencoded_Y)
        fi_results = fv.importance

    if fi_method_name == 'featurevec_regressor':
        fv = FeatureVec(mode='regress', feature_names= range(0, X.shape[1]))
        fv.fit(X, Y)
        fi_results = fv.importance

    #TODO: Works only with 1D Y as it unlists
    if fi_method_name == 'R_firm_classifier':
        robjects.globalenv["X"] = X
        robjects.globalenv["Y"] = Y
        r_return = robjects.r('''
        X <- as.data.frame(X)
        Y <- as.numeric(array(unlist(Y)))
        as_factor_Y <- as.factor(Y)
        ''')
        robjects.globalenv["method_tested"] = kwargs['method_name']
        fi_results = robjects.r('''
        fitControl <- trainControl(method = "none", classProbs = F)
        fitted_model <- train(x= X, y= as_factor_Y,
                method = method_tested,
                trControl = fitControl
                )
        Imp <- vi(fitted_model, method = "firm", scale = FALSE, sort = FALSE)
        as.vector(Imp$Importance)
        ''')
        fi_results = np.array(fi_results)

    #TODO: Works only with 1D Y as it unlists
    if fi_method_name == 'R_firm_regressor':
        robjects.globalenv["X"] = X
        robjects.globalenv["Y"] = Y
        r_return = robjects.r('''
        X <- as.data.frame(X)
        Y <- as.numeric(array(unlist(Y)))
        ''')
        robjects.globalenv["method_tested"] = kwargs['method_name']
        fi_results = robjects.r('''
        fitControl <- trainControl(method = "none", classProbs = F)
        fitted_model <- train(x= X, y= Y,
                method = method_tested,
                trControl = fitControl)
        Imp <- vi(fitted_model, method = "firm", scale = FALSE, sort = FALSE)
        as.vector(Imp$Importance)
        ''')
        fi_results = np.array(fi_results)



    # PCA
    if fi_method_name == 'PCA_sum':
        pca = PCA()
        X_new = pca.fit_transform(X)
        fi_results = np.sum(abs(pca.components_), axis = 0)

    # PCA_weighted same as PCA, but weighted by explained variance of component
    if fi_method_name == 'PCA_weighted':
        pca = PCA()
        X_new = pca.fit_transform(X)
        fi_results = np.sum(abs(pca.components_) * pca.explained_variance_ratio_, axis = 0)



    if fi_method_name == 'R_varimp_classifier':
        robjects.globalenv["X"] = X
        robjects.globalenv["Y"] = Y
        r_return = robjects.r('''
        X <- as.data.frame(X)
        Y <- as.numeric(array(unlist(Y)))
        as_factor_Y <- as.factor(Y)
        ''')
        fi_results = robjects.r('''
        data <- cbind(X, as_factor_Y)
        model <- cforest(as_factor_Y ~ ., data =  data)
        as.vector(varimp(model))
        ''')

    if fi_method_name == 'R_varimp_regressor':
        robjects.globalenv["X"] = X
        robjects.globalenv["Y"] = Y
        r_return = robjects.r('''
        X <- as.data.frame(X)
        Y <- as.numeric(array(unlist(Y)))
        ''')
        fi_results = robjects.r('''
        data <- cbind(X, Y)
        model <- cforest(Y ~ ., data =  data)
        as.vector(varimp(model))
        ''')

    if fi_method_name == 'R_pimp_classifier':
        robjects.globalenv["X"] = X
        robjects.globalenv["Y"] = Y
        r_return = robjects.r('''
        X <- as.data.frame(X)
        Y <- as.numeric(array(unlist(Y)))
        as_factor_Y <- as.factor(Y)
        ''')
        fi_results = robjects.r('''
        model_rf = randomForest(X,as_factor_Y,importance=TRUE)
        as.vector(PIMP(X,as_factor_Y,model_rf)$VarImp)
        ''')

    if fi_method_name == 'R_pimp_regressor':
        robjects.globalenv["X"] = X
        robjects.globalenv["Y"] = Y
        r_return = robjects.r('''
        X <- as.data.frame(X)
        Y <- as.numeric(array(unlist(Y)))
        ''')
        fi_results = robjects.r('''
        model_rf = randomForest(X,Y,importance=TRUE)
        as.vector(PIMP(X,Y,model_rf)$VarImp)
        ''')


    if fi_method_name == 'R_vip_sum_classifier':
        robjects.globalenv["X"] = X
        robjects.globalenv["Y"] = Y
        robjects.globalenv["method_tested"] = kwargs['method_name']
        fi_results = robjects.r('''
            Y <- as.factor(Y)
        model <- eval(parse(text= method_tested))(X, Y)
            linn.vip <- vip(model)
            rowSums(linn.vip)
            ''')

    if fi_method_name == 'R_vip_sum_regressor':
        robjects.globalenv["X"] = X
        robjects.globalenv["Y"] = Y
        robjects.globalenv["method_tested"] = kwargs['method_name']
        fi_results = robjects.r('''
        model <- eval(parse(text= method_tested))(X, Y)
            linn.vip <- vip(model)
            rowSums(linn.vip)
            ''')

    if fi_method_name == 'R_vip_weighted_X_classifier':
        robjects.globalenv["X"] = X
        robjects.globalenv["Y"] = Y
        robjects.globalenv["method_tested"] = kwargs['method_name']
        fi_results = robjects.r('''
        Y <- as.factor(Y)
        model <- eval(parse(text= method_tested))(X, Y)
            linn.vip <- vip(model)
            drop(t(linn.vip %*% model$prop_expl_var$X))
            ''')

    if fi_method_name == 'R_vip_weighted_X_regressor':
        robjects.globalenv["X"] = X
        robjects.globalenv["Y"] = Y
        robjects.globalenv["method_tested"] = kwargs['method_name']
        fi_results = robjects.r('''
        model <- eval(parse(text= method_tested))(X, Y)
            linn.vip <- vip(model)
            drop(t(linn.vip %*% model$prop_expl_var$X))
            ''')

    if fi_method_name == 'R_vip_weighted_Y_classifier':
        robjects.globalenv["X"] = X
        robjects.globalenv["Y"] = Y
        robjects.globalenv["method_tested"] = kwargs['method_name']
        fi_results = robjects.r('''
        Y <- as.factor(Y)
        model <- eval(parse(text= method_tested))(X, Y)
            linn.vip <- vip(model)
            drop(t(linn.vip %*% model$prop_expl_var$Y))
            ''')

    if fi_method_name == 'R_vip_weighted_Y_regressor':
        robjects.globalenv["X"] = X
        robjects.globalenv["Y"] = Y
        robjects.globalenv["method_tested"] = kwargs['method_name']
        fi_results = robjects.r('''
        model <- eval(parse(text= method_tested))(X, Y)
            linn.vip <- vip(model)
            drop(t(linn.vip %*% model$prop_expl_var$Y))
            ''')


    if fi_method_name == 'treeinterpreter_classifier':
        model = kwargs['model'].fit(X, labelencoded_Y)
        explainer = shap.explainers.Tree(model, X)
        contributions = treeinterpreter.predict(model, X)[2]
        average_abs_v = np.mean(np.abs(contributions), axis = 0)
        fi_results = convert_average_by_class(labelencoded_Y, X, model, average_abs_v)

    if fi_method_name == 'treeinterpreter_regressor':
        model = kwargs['model'].fit(X, Y)
        contributions = treeinterpreter.predict(model, X)[2]
        fi_results = np.mean(np.abs(contributions), axis = 0)





    return fi_results


# %%
# X, Y, labelencoded_Y, onehotencoded_Y, dataset = load_dataset(data_path='datasets/decimal_system.pickle')



# model = LinearRegression()
# model.fit(X,Y)
# loss = lambda x, y : np.power(x-y, 2)
# G = np.array([])
# res_1 = rfi(model.predict, loss, G, X, X, labelencoded_Y, np.arange(0, 3), n_repeats=30)
# print(res_1[0])

# res_2 = cfi(predict= model.predict, loss= loss, X_train= X, X_test= X, y_test= Y, D= np.arange(0, 3), n_repeats=30)
# print(res_2[0])

# for i in range(3):
#     print(res_1[0][i] / res_2[0][i])
# %%

# kwargs, fi_method_name = initialize_experiment_variables('Random_Forest_Classifier_gini')

# len(determine_fi(fi_method_name, data_path = 'E:/OneDrive/PhD/GitHub/Official_Feature_Importance/src/bp_feature_importance/datasets/decimal_system.pickle', **kwargs)) == 3



# np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
# for fi_method_name in method_list:
#     result = determine_fi(fi_method_name= fi_method_name, data_path= 'datasets/decimal_system.pickle')
#     print("{} gives fi: {}".format(fi_method_name, result))
# %%
# np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
# result_dict = {}
# X, Y, labelencoded_Y, onehotencoded_Y, dataset = load_dataset(data_path='datasets/decimal_system.pickle')

# for name in list_of_all_methods[-10:]:
#     kwargs, fi_method_name = initialize_experiment_variables(name, X, Y, labelencoded_Y)
#     print('\033[92m', name)
#     print('\033[94m')
#     result = determine_fi(fi_method_name=fi_method_name, data_path='datasets/decimal_system.pickle', **kwargs)
#     print("{} gives fi: {}".format(name, result))
#     result_dict[name] = result
# %%

# %%

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
result_dict = {}
X, Y, labelencoded_Y, onehotencoded_Y, dataset = load_dataset(data_path='datasets/decimal_system.pickle')
# %%
did_not_work = []

# test_methods = [
#  'permutation_importance_LogisticRegression',
#  'permutation_importance_SGDClassifier',
#  'permutation_importance_MLPClassifier',
#  'R_caret_classifier_treebag',
#  'R_caret_classifier_rpart1SE',
#  'R_caret_classifier_rpart2',
#  'R_caret_regressor_treebag',
#  'R_caret_regressor_earth',
#  'R_caret_regressor_bagEarth',
#  'R_caret_regressor_glmboost',
#  'R_caret_regressor_rpart1SE',
#  'R_caret_regressor_gcvEarth',
#  'R_caret_regressor_bagEarthGCV',
#  'R_caret_regressor_rpart2']

#test_methods = [i for i in list_of_all_methods if 'treeinterpreter' in i]
test_methods = list_of_all_methods
#test_methods = ['R_caret_classifier_pls',
 'R_firm_classifier_avNNet',
 'R_firm_classifier_pls']
for name in test_methods:
    kwargs, fi_method_name = initialize_experiment_variables(name, X, Y, labelencoded_Y)

    print('\033[94m')
    try:
        result = determine_fi(fi_method_name=fi_method_name, data_path='datasets/decimal_system.pickle', **kwargs)
        print("{} gives fi: {}".format(name, result))
        result_dict[name] = result
        assert len(result) == 3
        print('\033[92m', name)
        time.sleep(0.1)
    except:
        did_not_work += [name]
        print('\033[91m', name)
        time.sleep(0.1)
# %%



name = 'R_caret_classifier_pls'
kwargs, fi_method_name = initialize_experiment_variables(name, X, Y, labelencoded_Y)
result = determine_fi(fi_method_name=fi_method_name, data_path='datasets/decimal_system.pickle', **kwargs)
print(len(result))
# %%



for b in a:
    result = result_dict.get(b, [1])
    print(result)
    print(len(result))
# %%

# %%

test = importr('mixOmics')
# %%
robjects.globalenv["X"] = X
robjects.globalenv["Y"] = Y
robjects.r('''linn.pls <- pls(X, Y, ncomp = 3)
linn.vip <- vip(linn.pls)
rowSums(linn.vip)
drop(t(linn.vip %*% linn.pls$prop_expl_var$X))
drop(t(linn.vip %*% linn.pls$prop_expl_var$Y))
''')
# %%

# fit a scikit-learn's regressor model
rf = eval('RandomForestClassifier')()
rf.fit(X, labelencoded_Y)
# %%
prediction, bias, contributions = treeinterpreter.predict(rf, X)
# %%
prediction
