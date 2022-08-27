# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import math
from sklearn.metrics import mean_absolute_error
from collections import defaultdict
import glob
import os
from datetime import datetime
from method_lists import FAST_LIST, ALL_CLASSIFICATION
# %%
def change_name(name):
    listed_name = name
    if type(name) is not list:
        listed_name = [name]

    for ix, title in enumerate(listed_name):

        if title == 'independent_easy':
            title = 'Independence'

        if title == 'independent_hard':
            title = 'Fake independence'

        title = title.replace('_', ' ')
        listed_name[ix] = title

    if type(name) is not list:
        return listed_name[0]

    return listed_name

# %%
def group_together_results(target_path, path_names = [], results_after_path = None, save= False):


    if results_after_path is not None:
        with open(results_after_path, 'rb') as f:
            X, Y, labelencoded_Y, onehotencoded_Y, dataset, result_dict, time_dict, did_not_work, not_finished_in_time, time_limit, test_methods = pickle.load(f)

    elif results_after_path is None:
        did_not_work = []
        not_finished_in_time = []
        test_methods = []
        result_dict = {}
        time_dict = {}

    for path_name in path_names:
        # print(path_name)
        name = os.path.splitext(os.path.basename(path_name))[0].split('-', 1)[0]
        with open(path_name, 'rb') as f:
            result_dict[name], time_dict[name], X, Y, labelencoded_Y, onehotencoded_Y, dataset, _result_dict, _time_dict, _did_not_work, _not_finished_in_time, time_limit, _test_methods = pickle.load(f)
            did_not_work += _did_not_work
            not_finished_in_time += _not_finished_in_time
            test_methods += _test_methods



    did_not_work = list(set(did_not_work))
    not_finished_in_time = list(set(not_finished_in_time))
    test_methods = list(set(test_methods))

    if save == True:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path + f'-{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.pickle', 'wb') as f:
            pickle.dump([X, Y, labelencoded_Y, onehotencoded_Y, dataset, result_dict, time_dict, did_not_work, not_finished_in_time, time_limit, test_methods], f)
    return X, Y, labelencoded_Y, onehotencoded_Y, dataset, result_dict, time_dict, did_not_work, not_finished_in_time, time_limit, test_methods

# %%
class dataset_results:
    def __init__(self, path_name, data_name, method_list = None) -> None:
        self.path_name = path_name
        self.data_name = data_name
        self.method_list = method_list
        self.loss_function = mean_absolute_error
        self.load_results()
        self.check_results()
        self.remove_results()

    def load_results(self):
        # print(self.path_name)
        with open(self.path_name, 'rb') as f:
            self.X, self.Y, self.labelencoded_Y, self.onehotencoded_Y, self.dataset, self.result_dict, self.time_dict, self.did_not_work, self.not_finished_in_time, self.time_limit, self.test_methods = pickle.load(f)

        return

    def check_results(self):
        self.results_not_found = []
        if self.method_list is None:
            self.method_list = self.test_methods
        for i, method in enumerate(self.method_list):
            if method not in self.result_dict and method not in self.did_not_work and method not in self.not_finished_in_time:
                self.results_not_found += method
                print(f'{i}. {method} not found in {self.data_name}')


    def remove_results(self, extra_method_names = []):
        for method in self.did_not_work + self.not_finished_in_time + extra_method_names:
            try:
                del self.result_dict[method]
                del self.time_dict[method]
            except:
                pass
        return


# %%
class combined_dataset_results:
    def __init__(self, path_names, data_names, method_list = None) -> None:
        self.path_names = path_names
        self.data_names = data_names
        self.method_list = method_list
        self.results_obtained = {}

        try:
            os.remove('bp_results_text.txt')
        except:
            pass

        for path_name, data_name in zip(path_names, data_names):
            self.results_obtained[path_name] = dataset_results(path_name, data_name, self.method_list)





# %%

data_paths= os.listdir(f"datasets/final_datasets")
data_paths.remove('backup')
dataset_names = [os.path.basename(i).split('.pickle')[0] for i in data_paths]

path_names = []
data_names = []

should_group_first = False

for data_name in dataset_names:
    if should_group_first:
        group_together_results(target_path= f'results/final_results/{data_name}/grouped_results/grouped', path_names=glob.glob(f'results/final_results/{data_name}/*.pickle'), save = True)

    grouped_results = glob.glob(f'results/final_results/{data_name}/grouped_results/*.pickle')
    latest_file = max(grouped_results, key=os.path.getctime)

    path_names += [latest_file]
    data_names += [data_name]

# %%
results = combined_dataset_results(path_names, data_names, ALL_CLASSIFICATION)



# %%
# def check_properties(results):
    # per method, each property is checked for the datasets
    # 0 means counterexample found, 1 means not yet
properties_list = ['Efficiency', 'Symmetry', 'Range_0', 'Range_1', 'Bounds', 'Zero_FI', 'Indep_feat_not_imply_null_independent', 'Nul_independent_feat_imply_indep', 'Constant_var_zero_FI', 'FI_equal_to_one', 'Dep_1_not_imply_FI', 'Max_FI_when_fully_determined','Dep_1_gives_FI_bound', 'Limiting_outcome_space', 'Increasing_outcome_space', 'Not_subadditive_superadditive', 'Adding_features_can_increase_FI', 'Adding_feaures_can_decrease_FI', 'Cloning_does_not_increase_FI', 'Order_does_not_change_FI', 'XOR_dataset', 'Probability_dataset', 'Pairwise_dataset']

result_properties = {method : {prop : 1 for prop in properties_list} for method in results.method_list}

# Efficiency check:
if 'Efficiency' in properties_list:
    pass #!TODO

# Symmetry check:
if 'Symmetry' in properties_list:
    pass #!TODO

# Range_0 check:
if 'Range_0' in properties_list:
    for data_path,  data_results in results.results_obtained.items():
        # For every dataset:
        for method, FI_values in data_results.result_dict.items():
            if np.min(FI_values) < 0:
                result_properties[method]['Range_0'] = 0

# Range_1 check:
if 'Range_1' in properties_list:
    for data_path,  data_results in results.results_obtained.items():
        # For every dataset:
        for method, FI_values in data_results.result_dict.items():
            if np.min(FI_values) > 1:
                result_properties[method]['Range_1'] = 0


# Bounds check:
if 'Bounds' in properties_list:
    pass #!TODO

# Zero_FI check:
if 'Zero_FI' in properties_list:
    pass #!TODO

# Indep_feat_not_imply_null_independent check:
if 'Indep_feat_not_imply_null_independent' in properties_list:
    pass #!TODO

# Nul_independent_feat_imply_indep check:
if 'Nul_independent_feat_imply_indep' in properties_list:
    pass #!TODO

# Constant_var_zero_FI check:
if 'Constant_var_zero_FI' in properties_list:
    pass #!TODO

# FI_equal_to_one check:
if 'FI_equal_to_one' in properties_list:
    pass #!TODO

# Dep_1_not_imply_FI check:
if 'Dep_1_not_imply_FI' in properties_list:
    pass #!TODO

# Max_FI_when_fully_determined check:
if 'Max_FI_when_fully_determined' in properties_list:
    pass #!TODO

# Dep_1_gives_FI_bound check:
if 'Dep_1_gives_FI_bound' in properties_list:
    pass #!TODO

# Limiting_outcome_space check:
if 'Limiting_outcome_space' in properties_list:
    pass #!TODO

# Increasing_outcome_space check:
if 'Increasing_outcome_space' in properties_list:
    pass #!TODO

# Not_subadditive_superadditive check:
if 'Not_subadditive_superadditive' in properties_list:
    pass #!TODO

# Adding_features_can_increase_FI check:
if 'Adding_features_can_increase_FI' in properties_list:
    pass #!TODO

# Adding_feaures_can_decrease_FI check:
if 'Adding_feaures_can_decrease_FI' in properties_list:
    pass #!TODO

# Cloning_does_not_increase_FI check:
if 'Cloning_does_not_increase_FI' in properties_list:
    pass #!TODO

# Order_does_not_change_FI check:
if 'Order_does_not_change_FI' in properties_list:
    pass #!TODO

# XOR_dataset check:
if 'XOR_dataset' in properties_list:
    pass #!TODO

# Probability_dataset check:
if 'Probability_dataset' in properties_list:
    pass #!TODO

# Pairwise_dataset check:
if 'Pairwise_dataset' in properties_list:
    pass #!TODO


# %%

results.results_obtained['']