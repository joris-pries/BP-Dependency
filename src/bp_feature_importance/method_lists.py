standard_classification_models = ['LogisticRegression', 'Ridge', 'LinearRegression', 'Lasso', 'SGDClassifier', 'MLPClassifier', 'KNeighborsClassifier', 'GradientBoostingClassifier', 'AdaBoostClassifier', 'GaussianNB', 'BernoulliNB', 'LinearDiscriminantAnalysis', 'DecisionTreeClassifier', 'RandomForestClassifier', 'SVC', 'CatBoostClassifier', 'LGBMClassifier', 'XGBClassifier', 'XGBRFClassifier']

tree_based_classification_models = ['XGBClassifier', 'XGBRFClassifier', 'GradientBoostingClassifier', 'AdaBoostClassifier', 'DecisionTreeClassifier', 'RandomForestClassifier', 'CatBoostClassifier', 'LGBMClassifier', 'ExtraTreeClassifier', 'ExtraTreesClassifier']


# List of FI (classification only) methods:
ALL_CLASSIFICATION = []
# bp_feature_importance
ALL_CLASSIFICATION += ['bp_feature_importance']
# from scikit-learn
# AdaBoost_Classifier (default)
ALL_CLASSIFICATION += ['AdaBoost_Classifier']
# Random_Forest_Classifier (gini, entropy)
ALL_CLASSIFICATION += ['Random_Forest_Classifier_' + i for i in ['gini', 'entropy']]
# Extra_Trees_Classifier (gini, entropy)
ALL_CLASSIFICATION += ['Extra_Trees_Classifier_' + i for i in ['gini', 'entropy']]
# Gradient_Boosting_Classifier (default)
ALL_CLASSIFICATION += ['Gradient_Boosting_Classifier']
# KL_divergence (default)
ALL_CLASSIFICATION += ['KL_divergence']
# R_Mutual_Information (default)
ALL_CLASSIFICATION += ['R_Mutual_Information']
# SVR_absolute_weights (rbf, linear)
ALL_CLASSIFICATION += ['SVR_absolute_weights_' + i for i in ['linear']]
# EL_absolute_weights (default)
ALL_CLASSIFICATION += ['EL_absolute_weights']
# Fisher_Score (default)
ALL_CLASSIFICATION += ['Fisher_Score']
# permutation_importance_classifier
ALL_CLASSIFICATION += ['permutation_importance_classifier_' + i for i in standard_classification_models]
# shap_explainer_tree_classifier
ALL_CLASSIFICATION += ['shap_explainer_tree_classifier_' + i for i in tree_based_classification_models]
# shap_explainer_linear_classifier (LogisticRegression, SGDClassifier)
ALL_CLASSIFICATION += ['shap_explainer_linear_classifier_' + i for i in ['LogisticRegression', 'SGDClassifier']]
# shap_explainer_permutation_classifier (LogisticRegression, SGDClassifier, MLPClassifier, XGBClassifier, XGBRFClassifier)
ALL_CLASSIFICATION += ['shap_explainer_permutation_classifier_' +
                        i for i in standard_classification_models]

# shap_explainer_partition_classifier (LogisticRegression, SGDClassifier, MLPClassifier, XGBClassifier, XGBRFClassifier)
ALL_CLASSIFICATION += ['shap_explainer_partition_classifier_' + i for i in standard_classification_models]

# shap_explainer_sampling_classifier (LogisticRegression, SGDClassifier, MLPClassifier, XGBClassifier, XGBRFClassifier)
ALL_CLASSIFICATION += ['shap_explainer_sampling_classifier_' + i for i in standard_classification_models]

# shap_explainer_kernel_classifier (LogisticRegression, SGDClassifier, MLPClassifier, XGBClassifier, XGBRFClassifier)
ALL_CLASSIFICATION += ['shap_explainer_kernel_classifier_' + i for i in standard_classification_models]

# shap_explainer_exact_classifier (LogisticRegression, SGDClassifier, MLPClassifier, XGBClassifier, XGBRFClassifier)
ALL_CLASSIFICATION += ['shap_explainer_exact_classifier_' + i for i in standard_classification_models]

# R_caret_classifier
ALL_CLASSIFICATION += ['R_caret_classifier_' + i for i in
                        ['snn', 'knn', 'bayesglm', 'lssvmRadial', 'rocc', 'ownn', 'ORFpls', 'rFerns', 'treebag', 'RRF', 'svmRadial', 'ctree2', 'evtree', 'pda', 'rpart', 'cforest', 'svmLinear', 'xyf', 'C5.0Tree', 'avNNet', 'kknn', 'svmRadialCost', 'gaussprRadial', 'FH.GBML', 'svmLinear2', 'bstSm', 'LogitBoost', 'wsrf', 'plr', 'xgbLinear', 'rf', 'null', 'protoclass', 'monmlp', 'Rborist', 'mlpWeightDecay', 'svmRadialWeights', 'mlpML', 'ctree', 'loclda', 'sdwd', 'mlpWeightDecayML', 'svmRadialSigma', 'bstTree', 'dnn', 'ordinalRF', 'pda2', 'BstLm', 'RRFglobal', 'mlp', 'rpart1SE', 'pcaNNet', 'ORFsvm', 'parRF', 'rpart2', 'gaussprPoly', 'C5.0Rules', 'rda', 'rbfDDA', 'multinom', 'gaussprLinear', 'svmPoly']]

# rfi_classifier
ALL_CLASSIFICATION += ['rfi_classifier_' + i for i in standard_classification_models]

# cfi_classifier
ALL_CLASSIFICATION += ['cfi_classifier_' + i for i in standard_classification_models]

# featurevec_classifier
ALL_CLASSIFICATION += ['featurevec_classifier']

# R_firm_classifier
ALL_CLASSIFICATION += ['R_firm_classifier_' + i for i in
                        ['knn', 'treebag', 'RRF', 'ctree2', 'evtree', 'pda', 'rpart', 'cforest', 'xyf', 'C5.0Tree', 'kknn', 'gaussprRadial', 'LogitBoost', 'wsrf', 'xgbLinear', 'rf', 'null', 'monmlp', 'Rborist', 'mlpWeightDecay', 'mlpML', 'ctree', 'mlpWeightDecayML', 'dnn', 'pda2', 'RRFglobal', 'mlp', 'rpart1SE', 'parRF', 'rpart2', 'gaussprPoly', 'C5.0Rules', 'rbfDDA', 'multinom', 'gaussprLinear']]

# PCA
ALL_CLASSIFICATION += ['PCA_sum']
# PCA_weighted
ALL_CLASSIFICATION += ['PCA_weighted']
# varimp_classifier
ALL_CLASSIFICATION += ['R_varimp_classifier']
# pimp_classifier
ALL_CLASSIFICATION += ['R_pimp_classifier']
# vip_sum_classifier
ALL_CLASSIFICATION += ['R_vip_sum_classifier_' + i for i in ['plsda', 'splsda']]
# vip_weighted_X_classifier
ALL_CLASSIFICATION += ['R_vip_weighted_X_classifier_' + i for i in ['plsda', 'splsda']]
# vip_weighted_Y_classifier
ALL_CLASSIFICATION += ['R_vip_weighted_Y_classifier_' + i for i in ['plsda', 'splsda']]
# treeinterpreter_classifier
ALL_CLASSIFICATION += ['treeinterpreter_classifier_' + i for i in tree_based_classification_models]

# scipy_stats
ALL_CLASSIFICATION += ['scipy_stats_' + i for i in ['f_oneway', 'alexandergovern', 'pearsonr', 'spearmanr', 'pointbiserialr',
                                                     'kendalltau', 'weightedtau', 'somersd', 'linregress', 'siegelslopes', 'theilslopes', 'multiscale_graphcorr']]
# booster_classifier (XGBClassifier)
ALL_CLASSIFICATION += ['booster_classifier_' + j + '_' + i for j in ['weight', 'gain', 'cover'] for i in tree_based_classification_models]

# TODO uitzoeken of chi2 regression of classification is
# sklearn_feature_selection_classifier
ALL_CLASSIFICATION += ['sklearn_feature_selection_classifier_' + i for i in ['chi2', 'f_classif', 'mutual_info_classif']]

# R_FSinR_classifier
ALL_CLASSIFICATION += ['R_FSinR_classifier_' + i for i in ['binaryConsistency', 'chiSquared', 'cramer', 'gainRatio', 'giniIndex',
                                                            'IEConsistency', 'IEPConsistency', 'mutualInformation',  'roughsetConsistency', 'ReliefFeatureSetMeasure', 'symmetricalUncertain']]

# sage_classifier
ALL_CLASSIFICATION += ['sage_classifier_' + j + '_' + i for j in ['IteratedEstimator', 'PermutationEstimator', 'KernelEstimator', 'SignEstimator']
                        for i in standard_classification_models]


# QII_averaged_classifier
ALL_CLASSIFICATION += ['QII_averaged_classifier_' + j + '_' + i for j in ['shapley', 'banzhaf']
                        for i in standard_classification_models]

# sunnies
ALL_CLASSIFICATION += ['sunnies_' + i for i in ['R2', 'DC', 'BCDC', 'AIDC', 'HSIC']]
# rebelosa_classifier
ALL_CLASSIFICATION += ['rebelosa_classifier_' + i for i in ['RF', 'Garson_NN1', 'Garson_NN2', 'VIANN_NN1', 'VIANN_NN2', 'LOFO_NN1', 'LOFO_NN2']]

# relief_classifier
ALL_CLASSIFICATION += ['relief_classifier_' + i for i in ['Relief', 'ReliefF', 'RReliefF']]

# DIFFI
ALL_CLASSIFICATION += ['DIFFI']
# ITMO
ALL_CLASSIFICATION += ['ITMO_' + i for i in ["fit_criterion_measure", "f_ratio_measure", "gini_index", "su_measure", "spearman_corr", "pearson_corr", "fechner_corr", "kendall_corr", "chi2_measure", "anova", "laplacian_score",
                                              "information_gain", "modified_t_score"] + ["MIM", "MRMR", "JMI", "CIFE", "CMIM", "ICAP", "DCSF", "CFR", "MRI", "IWFS"] + ['NDFS', 'RFS', 'SPEC', 'MCFS', 'UDFS']]

















FAST_LIST = ['sage_classifier_SignEstimator_Ridge',
'QII_averaged_classifier_banzhaf_LinearDiscriminantAnalysis',
'sage_classifier_IteratedEstimator_SVC',
'sage_classifier_IteratedEstimator_XGBClassifier',
'booster_classifier_weight_DecisionTreeClassifier',
'sage_classifier_SignEstimator_Lasso',
'sage_classifier_SignEstimator_SVC',
'booster_classifier_gain_DecisionTreeClassifier',
'sage_classifier_PermutationEstimator_SGDClassifier',
'treeinterpreter_classifier_XGBClassifier',
'R_caret_classifier_gaussprRadial',
'shap_explainer_sampling_classifier_LinearDiscriminantAnalysis',
'sage_classifier_PermutationEstimator_Ridge',
'booster_classifier_cover_ExtraTreesClassifier',
'sage_classifier_IteratedEstimator_LinearDiscriminantAnalysis',
'booster_classifier_cover_AdaBoostClassifier',
'QII_averaged_classifier_shapley_LinearDiscriminantAnalysis',
'sage_classifier_SignEstimator_AdaBoostClassifier',
'treeinterpreter_classifier_AdaBoostClassifier',
'treeinterpreter_classifier_ExtraTreesClassifier',
'shap_explainer_tree_classifier_ExtraTreeClassifier',
'treeinterpreter_classifier_ExtraTreeClassifier',
'sage_classifier_KernelEstimator_XGBClassifier',
'booster_classifier_cover_RandomForestClassifier',
'sage_classifier_IteratedEstimator_SGDClassifier',
'sage_classifier_SignEstimator_LinearRegression',
'sage_classifier_IteratedEstimator_Lasso',
'rfi_classifier_LinearDiscriminantAnalysis',
'sage_classifier_SignEstimator_XGBClassifier',
'shap_explainer_tree_classifier_XGBRFClassifier',
'R_caret_classifier_rpart1SE',
'sage_classifier_PermutationEstimator_XGBClassifier',
'R_caret_classifier_pcaNNet',
'R_firm_classifier_cforest',
'shap_explainer_permutation_classifier_LinearDiscriminantAnalysis',
'booster_classifier_cover_DecisionTreeClassifier',
'shap_explainer_linear_classifier_LogisticRegression',
'booster_classifier_gain_RandomForestClassifier',
'sage_classifier_PermutationEstimator_Lasso',
'sage_classifier_PermutationEstimator_LinearDiscriminantAnalysis',
'sage_classifier_SignEstimator_GaussianNB',
'ITMO_MCFS',
'sage_classifier_KernelEstimator_XGBRFClassifier',
'booster_classifier_cover_LGBMClassifier',
'shap_explainer_tree_classifier_AdaBoostClassifier',
'sage_classifier_SignEstimator_SGDClassifier',
'booster_classifier_gain_ExtraTreeClassifier',
'booster_classifier_cover_GradientBoostingClassifier',
'cfi_classifier_LinearDiscriminantAnalysis',
'sage_classifier_IteratedEstimator_XGBRFClassifier',
'sage_classifier_SignEstimator_LinearDiscriminantAnalysis',
'permutation_importance_classifier_LinearDiscriminantAnalysis',
'treeinterpreter_classifier_DecisionTreeClassifier',
'treeinterpreter_classifier_RandomForestClassifier',
'R_caret_classifier_multinom',
'sage_classifier_KernelEstimator_SVC',
'booster_classifier_gain_ExtraTreesClassifier',
'shap_explainer_tree_classifier_LGBMClassifier',
'shap_explainer_linear_classifier_SGDClassifier',
'booster_classifier_weight_ExtraTreeClassifier',
'sage_classifier_PermutationEstimator_XGBRFClassifier',
'booster_classifier_gain_AdaBoostClassifier',
'sage_classifier_KernelEstimator_Ridge',
'treeinterpreter_classifier_LGBMClassifier',
'sage_classifier_PermutationEstimator_LinearRegression',
'shap_explainer_kernel_classifier_LinearDiscriminantAnalysis',
'sage_classifier_IteratedEstimator_Ridge',
'sage_classifier_SignEstimator_XGBRFClassifier',
'sage_classifier_KernelEstimator_LinearRegression',
'booster_classifier_cover_ExtraTreeClassifier',
'R_caret_classifier_FH.GBML',
'sage_classifier_IteratedEstimator_DecisionTreeClassifier',
'booster_classifier_weight_LGBMClassifier',
'sage_classifier_IteratedEstimator_LinearRegression',
'R_firm_classifier_evtree',
'shap_explainer_exact_classifier_LinearDiscriminantAnalysis',
'sage_classifier_KernelEstimator_LinearDiscriminantAnalysis',
'shap_explainer_tree_classifier_XGBClassifier',
'sage_classifier_KernelEstimator_SGDClassifier',
'booster_classifier_gain_GradientBoostingClassifier',
'sage_classifier_IteratedEstimator_GaussianNB',
'booster_classifier_weight_AdaBoostClassifier',
'sage_classifier_IteratedEstimator_BernoulliNB',
'sage_classifier_KernelEstimator_Lasso',
'booster_classifier_gain_LGBMClassifier',
'shap_explainer_partition_classifier_LinearDiscriminantAnalysis',
'sage_classifier_PermutationEstimator_SVC',
'treeinterpreter_classifier_GradientBoostingClassifier',
'treeinterpreter_classifier_XGBRFClassifier',
'R_caret_classifier_rpart2',
'booster_classifier_weight_ExtraTreesClassifier',
'booster_classifier_weight_GradientBoostingClassifier',
'sage_classifier_SignEstimator_BernoulliNB',
'shap_explainer_tree_classifier_DecisionTreeClassifier',
'sage_classifier_IteratedEstimator_RandomForestClassifier',
'R_firm_classifier_multinom',
'shap_explainer_tree_classifier_GradientBoostingClassifier',
'booster_classifier_weight_RandomForestClassifier',
'sage_classifier_SignEstimator_DecisionTreeClassifier',
'ITMO_CFR',
'ITMO_DCSF',
'ITMO_pearson_corr',
'scipy_stats_pearsonr',
'ITMO_IWFS',
'ITMO_fechner_corr',
'scipy_stats_pointbiserialr',
'KL_divergence',
'scipy_stats_linregress',
'EL_absolute_weights',
'ITMO_gini_index',
'PCA_weighted',
'scipy_stats_alexandergovern',
'scipy_stats_spearmanr',
'ITMO_spearman_corr',
'scipy_stats_kendalltau',
'ITMO_modified_t_score',
'sklearn_feature_selection_classifier_chi2',
'permutation_importance_classifier_Ridge',
'permutation_importance_classifier_Lasso',
'R_FSinR_classifier_chiSquared',
'permutation_importance_classifier_LinearRegression',
'R_FSinR_classifier_cramer',
'R_Mutual_Information',
'ITMO_information_gain',
'sklearn_feature_selection_classifier_f_classif',
'ITMO_MRI',
'R_FSinR_classifier_symmetricalUncertain',
'ITMO_MIM',
'relief_classifier_Relief',
'R_FSinR_classifier_gainRatio',
'ITMO_CIFE',
'R_FSinR_classifier_mutualInformation',
'ITMO_CMIM',
'Fisher_Score',
'ITMO_JMI',
'ITMO_anova',
'scipy_stats_weightedtau',
'SVR_absolute_weights_linear',
'ITMO_MRMR',
'ITMO_su_measure',
'ITMO_ICAP',
'PCA_sum',
'cfi_classifier_Ridge',
'cfi_classifier_LinearRegression',
'cfi_classifier_Lasso',
'rfi_classifier_LinearRegression',
'ITMO_laplacian_score',
'rfi_classifier_Lasso',
'R_vip_weighted_X_classifier_splsda',
'R_vip_weighted_Y_classifier_plsda',
'R_vip_sum_classifier_splsda',
'scipy_stats_somersd',
'rfi_classifier_Ridge',
'ITMO_chi2_measure',
'R_vip_sum_classifier_plsda',
'ITMO_f_ratio_measure',
'permutation_importance_classifier_DecisionTreeClassifier',
'R_FSinR_classifier_IEConsistency',
'R_FSinR_classifier_roughsetConsistency',
'R_FSinR_classifier_giniIndex',
'R_vip_weighted_Y_classifier_splsda',
'DIFFI',
'R_vip_weighted_X_classifier_plsda',
'permutation_importance_classifier_BernoulliNB',
'ITMO_kendall_corr',
'R_FSinR_classifier_IEPConsistency',
'R_caret_classifier_C5.0Rules',
'scipy_stats_siegelslopes',
'Random_Forest_Classifier_gini',
'Extra_Trees_Classifier_gini',
'Random_Forest_Classifier_entropy',
'permutation_importance_classifier_SGDClassifier',
'R_FSinR_classifier_binaryConsistency',
'scipy_stats_theilslopes',
'Extra_Trees_Classifier_entropy',
'scipy_stats_f_oneway',
'AdaBoost_Classifier',
'cfi_classifier_DecisionTreeClassifier',
'permutation_importance_classifier_LogisticRegression',
'permutation_importance_classifier_KNeighborsClassifier',
'sunnies_R2',
'cfi_classifier_BernoulliNB',
'featurevec_classifier',
'rfi_classifier_BernoulliNB',
'permutation_importance_classifier_GaussianNB',
'rfi_classifier_DecisionTreeClassifier',
'cfi_classifier_SGDClassifier',
'R_firm_classifier_rpart1SE',
'cfi_classifier_LogisticRegression',
'R_firm_classifier_rpart',
'ITMO_RFS',
'R_firm_classifier_null',
'R_caret_classifier_sdwd',
'R_caret_classifier_parRF',
'rfi_classifier_SGDClassifier',
'R_caret_classifier_rf',
'ITMO_SPEC',
'R_caret_classifier_ordinalRF',
'booster_classifier_weight_XGBClassifier',
'ITMO_fit_criterion_measure',
'R_firm_classifier_C5.0Tree',
'R_caret_classifier_RRFglobal',
'permutation_importance_classifier_XGBClassifier',
'rfi_classifier_LogisticRegression',
'shap_explainer_permutation_classifier_Ridge',
'shap_explainer_permutation_classifier_LinearRegression',
'shap_explainer_permutation_classifier_Lasso',
'permutation_importance_classifier_MLPClassifier',
'cfi_classifier_KNeighborsClassifier',
'permutation_importance_classifier_XGBRFClassifier',
'booster_classifier_weight_XGBRFClassifier',
'relief_classifier_RReliefF',
'rebelosa_classifier_RF',
'R_caret_classifier_xgbLinear',
'R_firm_classifier_kknn',
'R_firm_classifier_rpart2',
'rfi_classifier_GaussianNB',
'R_firm_classifier_rbfDDA',
'R_firm_classifier_mlpWeightDecay',
'booster_classifier_cover_XGBClassifier',
'booster_classifier_cover_XGBRFClassifier',
'rfi_classifier_KNeighborsClassifier',
'cfi_classifier_MLPClassifier',
'R_firm_classifier_mlpWeightDecayML',
'R_firm_classifier_dnn',
'shap_explainer_exact_classifier_LinearRegression',
'R_firm_classifier_mlpML',
'R_firm_classifier_mlp',
'cfi_classifier_GaussianNB',
'rfi_classifier_XGBClassifier',
'permutation_importance_classifier_LGBMClassifier',
'booster_classifier_gain_XGBClassifier',
'cfi_classifier_XGBClassifier',
'rfi_classifier_XGBRFClassifier',
'R_firm_classifier_xyf',
'R_firm_classifier_parRF',
'permutation_importance_classifier_RandomForestClassifier',
'R_firm_classifier_RRF',
'shap_explainer_exact_classifier_Lasso',
'R_caret_classifier_Rborist',
'R_firm_classifier_ctree2',
'R_firm_classifier_RRFglobal',
'cfi_classifier_XGBRFClassifier',
'R_firm_classifier_treebag',
'R_caret_classifier_RRF',
'R_firm_classifier_rf',
'rebelosa_classifier_Garson_NN2',
'rebelosa_classifier_VIANN_NN1',
'rebelosa_classifier_VIANN_NN2',
'rebelosa_classifier_LOFO_NN1',
'rebelosa_classifier_LOFO_NN2',
'rebelosa_classifier_Garson_NN1',
'R_firm_classifier_pda2',
'booster_classifier_gain_XGBRFClassifier',
'rfi_classifier_MLPClassifier',
'permutation_importance_classifier_AdaBoostClassifier',
'shap_explainer_partition_classifier_LinearRegression',
'shap_explainer_partition_classifier_Ridge',
'shap_explainer_partition_classifier_Lasso',
'R_caret_classifier_treebag',
'sunnies_AIDC',
'ITMO_UDFS',
'sunnies_DC',
'R_firm_classifier_Rborist',
'shap_explainer_partition_classifier_DecisionTreeClassifier',
'sunnies_BCDC',
'bp_feature_importance',
'relief_classifier_ReliefF',
'R_firm_classifier_LogitBoost',
'rfi_classifier_LGBMClassifier',
'R_firm_classifier_pda',
'cfi_classifier_LGBMClassifier',
'sklearn_feature_selection_classifier_mutual_info_classif',
'cfi_classifier_RandomForestClassifier',
'ITMO_NDFS',
'R_firm_classifier_xgbLinear',
'shap_explainer_permutation_classifier_SGDClassifier',
'shap_explainer_permutation_classifier_LogisticRegression',
'permutation_importance_classifier_SVC',
'shap_explainer_partition_classifier_LogisticRegression',
'rfi_classifier_RandomForestClassifier',
'R_caret_classifier_rbfDDA',
'R_caret_classifier_rda',
'shap_explainer_permutation_classifier_DecisionTreeClassifier',
'R_caret_classifier_mlp',
'R_caret_classifier_ORFsvm',
'R_caret_classifier_BstLm',
'R_caret_classifier_pda2',
'cfi_classifier_AdaBoostClassifier',
'R_caret_classifier_knn',
'R_caret_classifier_snn',
'shap_explainer_sampling_classifier_DecisionTreeClassifier',
'R_caret_classifier_bayesglm',
'R_firm_classifier_wsrf',
'shap_explainer_sampling_classifier_Lasso',
'sage_classifier_IteratedEstimator_LogisticRegression',
'sage_classifier_IteratedEstimator_MLPClassifier',
'R_caret_classifier_plr',
'shap_explainer_sampling_classifier_Ridge',
'R_caret_classifier_null',
'shap_explainer_sampling_classifier_BernoulliNB',
'R_caret_classifier_lssvmRadial',
'R_caret_classifier_protoclass',
'shap_explainer_sampling_classifier_LinearRegression',
'sage_classifier_IteratedEstimator_KNeighborsClassifier',
'shap_explainer_exact_classifier_DecisionTreeClassifier',
'R_caret_classifier_dnn',
'shap_explainer_sampling_classifier_KNeighborsClassifier',
'rfi_classifier_AdaBoostClassifier',
'R_caret_classifier_ownn',
'R_caret_classifier_rocc',
'R_caret_classifier_rFerns',
'R_caret_classifier_mlpWeightDecay',
'shap_explainer_permutation_classifier_MLPClassifier',
'R_caret_classifier_ORFpls',
'shap_explainer_partition_classifier_MLPClassifier',
'R_firm_classifier_monmlp',
'shap_explainer_sampling_classifier_LogisticRegression',
'R_FSinR_classifier_ReliefFeatureSetMeasure',
'shap_explainer_partition_classifier_SGDClassifier',
'shap_explainer_permutation_classifier_BernoulliNB',
'shap_explainer_partition_classifier_BernoulliNB',
'R_caret_classifier_monmlp',
'R_caret_classifier_mlpWeightDecayML',
'R_pimp_classifier',
'shap_explainer_permutation_classifier_KNeighborsClassifier',
'shap_explainer_sampling_classifier_MLPClassifier',
'shap_explainer_exact_classifier_Ridge',
'shap_explainer_exact_classifier_BernoulliNB',
'shap_explainer_sampling_classifier_GaussianNB',
'shap_explainer_exact_classifier_SGDClassifier',
'Gradient_Boosting_Classifier',
'R_caret_classifier_loclda',
'shap_explainer_sampling_classifier_XGBClassifier',
'shap_explainer_sampling_classifier_SGDClassifier',
'shap_explainer_partition_classifier_XGBRFClassifier',
'shap_explainer_sampling_classifier_XGBRFClassifier',
'permutation_importance_classifier_GradientBoostingClassifier',
'sage_classifier_SignEstimator_LogisticRegression',
'R_firm_classifier_ctree',
'R_firm_classifier_C5.0Rules',
'shap_explainer_partition_classifier_KNeighborsClassifier',
'sage_classifier_SignEstimator_KNeighborsClassifier',
'R_caret_classifier_wsrf',
'sage_classifier_SignEstimator_MLPClassifier',
'R_caret_classifier_ctree',
'shap_explainer_exact_classifier_XGBClassifier',
'R_caret_classifier_mlpML',
'cfi_classifier_GradientBoostingClassifier',
'shap_explainer_exact_classifier_XGBRFClassifier',
'sage_classifier_PermutationEstimator_KNeighborsClassifier',
'shap_explainer_exact_classifier_MLPClassifier',
'R_caret_classifier_LogitBoost',
'shap_explainer_sampling_classifier_RandomForestClassifier',
'shap_explainer_partition_classifier_XGBClassifier',
'R_firm_classifier_knn'
]

