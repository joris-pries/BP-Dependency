# %%
import numpy as np
#import pandas as pd
import random
import itertools
import math
import cProfile
import pstats

import sys
sys.path.insert(0, 'E:/OneDrive/PhD/GitHub/Official_Dependency_Function/src')

from bp_dependency import *
#from pyparsing import col


#__all__ = ['bin_data', 'unordered_bp_dependency', 'bp_dependency']


def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])

def flatten_and_np_array(list_of_lists):
    return np.asarray(flatten(list_of_lists))

# %%
n_observations = 20000000

# X_1 = np.random.uniform(low = 0.0, high= 1.0, size = n_observations)
# X_2 = np.random.uniform(low = 0.0, high= 1.0, size = n_observations)

# Y = X_1 + 3 * X_2

# X_indices= [0,1]
# Y_indices = [2]
# binning_indices = [0,1,2]
# dataset = np.stack((X_1, X_2, Y), axis = 1)

n_x_variables = 5
X = np.random.randint(2, size=(n_observations, n_x_variables))
Y_random_index = random.choices(range(n_x_variables), weights=[0.3, 0.05, 0.3, 0.15, 0.2], k= n_observations)
Y = np.asarray([X[i, Y_random_index[i]] for i in range(n_observations)]).reshape((n_observations, 1))
# %%
S_0 = np.asarray([int(i == 0) for i in Y_random_index]).reshape((n_observations, 1))
S_1 = np.asarray([int(i == 1) for i in Y_random_index]).reshape((n_observations, 1))
S_2 = np.asarray([int(i == 2) for i in Y_random_index]).reshape((n_observations, 1))
S_3 = np.asarray([int(i == 3) for i in Y_random_index]).reshape((n_observations, 1))
S_4 = np.asarray([int(i == 4) for i in Y_random_index]).reshape((n_observations, 1))


dataset = np.hstack((X,Y,S_0,S_1,S_2,S_3,S_4))
X_indices = [[0,6], [1,7], [2,8],[3,9],[4,10]]
Y_indices = [5]

# %%

# X_1 = np.random.randint(2, size=(n_observations, 1))
# X_2 = np.random.randint(low = 2, high = 4, size=(n_observations, 1))
# Y_random_index = np.asarray(random.choices(range(2), weights=[0.3, 0.7], k= n_observations))
# Y = np.zeros(shape= (n_observations, 1))
# for i in range(n_observations):
#     Y[i,0] = X_1[i,0]
#     if Y_random_index[i] == 1:
#         Y[i,0] = X_2[i,0]


# Y_random_index = Y_random_index[..., np.newaxis]
# dataset = np.hstack((X_1, X_2, Y_random_index,Y))
# X_indices = [0,1,2]
# Y_indices = [3]

# %%

# S = np.random.randint(4, size = n_observations)
# help_data = np.array([[0, 0], [1, 1], [2, 0], [3, 1]])
# Y = np.array([help_data[S[i], 0] for i in range(n_observations)])
# X =  np.array([help_data[S[i], 1] for i in range(n_observations)])
# dataset = np.vstack((Y,X)).T
# X_indices = [1]
# Y_indices = [0]
# %%



class feature_importance_class:



    def __init__(self, dataset, X_indices, Y_indices, stopping_strategy, sequence_strategy, epsilon, limit_n_variables, binning_indices, binning_strategy, midway_binning) -> None:
        # Input variables
        self.dataset = np.asarray(dataset.copy())
        self.X_indices = X_indices
        self.X_indices = [X_index if isinstance(X_index, list) else [X_index] for X_index in X_indices]
        self.Y_indices = np.asarray(Y_indices)
        # self.grouped_X_indices = list(grouped_X_indices)
        # self.grouped_X_indices_set = {elem[0]: elem for elem in grouped_X_indices}
        # flattened_grouped_X_indices = [item for sublist in grouped_X_indices for item in sublist]
        # help_set = {X_index: [X_index] for X_index in X_indices if X_index not in flattened_grouped_X_indices} 
        # self.grouped_X_indices_set.update(help_set)
        # self.reduced_X_indices = self.grouped_X_indices_set.keys()
        
        self.stopping_strategy = stopping_strategy
        self.sequence_strategy = sequence_strategy
        self.epsilon = epsilon
        self.limit_n_variables = limit_n_variables
        self.binning_indices = binning_indices
        if self.binning_indices is not None:
            self.binning_indices = np.asarray(self.binning_indices)
        self.binning_strategy = binning_strategy
        if self.binning_indices is not None:
            if not isinstance(self.binning_strategy, dict):
                strat = self.binning_strategy
                if self.binning_indices.ndim == 0:
                    self.binning_strategy = {binning_indices : strat}
                else:
                    self.binning_strategy = {bin_index : strat for bin_index in self.binning_indices}
        self.midway_binning = midway_binning

        # Binning the data
        if self.binning_indices is not None:
            for bin_index in binning_indices:
                self.dataset[:, bin_index] = bin_data(x= self.dataset[:, bin_index], bins= self.binning_strategy[bin_index], midways= self.midway_binning)

        # Dependency specific:
        self.UD_Y_Y = unordered_bp_dependency(dataset= self.dataset, X_indices= self.Y_indices, Y_indices= self.Y_indices, binning_indices= None, format_input= False)
        self.UD_all_X_Y = unordered_bp_dependency(dataset= self.dataset, X_indices= flatten_and_np_array(self.X_indices), Y_indices= self.Y_indices, binning_indices= None, format_input= False)
        self.UD_before = 0
        self.UD_after = 0

        #TODO: Ergens checken of dingen niet constant zijn

        # Automatic variables
        self.n_sequences_counter  = -1
        self.total_sequence = None
        self.current_sequence = None
        self.current_sequence_variable = None
        self.current_sequence_counter = -1


        self.new_ud_value = None
        self.average_shapley_values = {frozenset(index) : 0 for index in self.X_indices}
        self.average_shapley_values_counters = {frozenset(index) : 0 for index in self.X_indices}
        # dict that keeps track of which ud dependencies have already been computed
        self.computed_ud_dependencies = {frozenset(flatten_and_np_array(self.X_indices)) : self.UD_all_X_Y}


        # strategy specific
        if self.sequence_strategy == 'exhaustive':        
            self.sequence_array = list(itertools.permutations(self.X_indices))
            self.stopping_strategy = math.factorial(len(self.X_indices))

    # Functie die bepaald of er gestopt moet worden met het hele proces
    def stop_generating_sequences(self) -> bool:
        if isinstance(self.stopping_strategy, int):
            # In this case, we want to generate exactly stopping_strategy sequences (that is why -1)
            return(self.n_sequences_counter >= self.stopping_strategy - 1)






    # Functie die bepaald of er gestopt moet worden met 1 sequence
    def early_sequence_stopping(self) -> bool:
        if self.UD_all_X_Y - self.UD_after <= self.epsilon * self.UD_Y_Y:
            return(True)
        if len(self.current_sequence) >= self.limit_n_variables:
            return(True)
        
        return(False)



    # Functie die gemiddelde shapley values bijhoudt en bijvoorbeeld hoe vaak ze gekozen zijn om de gemiddeldes uit te rekenen
    def update_shapley_value(self):
        self.average_shapley_values[frozenset(self.current_sequence_variable)] = (self.average_shapley_values_counters[frozenset(self.current_sequence_variable)] * self.average_shapley_values[frozenset(self.current_sequence_variable)] + self.new_ud_value) / (self.average_shapley_values_counters[frozenset(self.current_sequence_variable)] + 1)
        self.average_shapley_values_counters[frozenset(self.current_sequence_variable)] += 1



    # Functie die toegevoegde waardes bepaald:
    def determine_shapley_value(self):

        try:
            self.UD_after = self.computed_ud_dependencies[frozenset(flatten_and_np_array(self.current_sequence))]
        except:
            self.UD_after = unordered_bp_dependency(dataset= self.dataset, X_indices= flatten_and_np_array(self.current_sequence), Y_indices= self.Y_indices, binning_indices= None, format_input= False)      
            # Update computed dependecies dict
            self.computed_ud_dependencies[frozenset(flatten_and_np_array(self.current_sequence))] = self.UD_after
        

        self.new_ud_value = self.UD_after - self.UD_before

        # For next time, update UD_before
        self.UD_before = self.UD_after




    # Functie die shapley sequences genereert
    def generate_shapley_sequence(self):
        self.n_sequences_counter += 1

        # Gewoon willekeurig
        if self.sequence_strategy == 'random':
            index_list = self.X_indices.copy()
            random.shuffle(index_list)
            self.total_sequence = index_list

        # Gebruikt een lijst om die 1 voor 1 af te lopen
        if self.sequence_strategy == 'exhaustive':
            return_sequence = list(self.sequence_array[self.n_sequences_counter])
            self.total_sequence = return_sequence


    def reset_after_sequence(self):
        self.current_sequence_counter = -1
        self.UD_after = 0
        self.UD_before = 0
        self.current_sequence = self.total_sequence[:1]
        self.current_sequence_variable = self.total_sequence[0]

    def next_variable_sequence(self):
        self.current_sequence_counter += 1
        self.current_sequence = self.total_sequence[: self.current_sequence_counter + 1]
        self.current_sequence_variable = self.total_sequence[self.current_sequence_counter]


    def divide_average_shapley_by_Y(self):
        self.average_shapley_values = {frozenset(key): value / self.UD_Y_Y for key,value in self.average_shapley_values.items()}



# %%



# %%
def bp_feature_importance(dataset, X_indices, Y_indices, stopping_strategy = 120, sequence_strategy= 'exhaustive', epsilon= 0.0, limit_n_variables= np.inf, binning_indices= None, binning_strategy = 'auto', midway_binning = False):

    bp_class = feature_importance_class(
        dataset= dataset,
        X_indices= X_indices,
        Y_indices= Y_indices,
        stopping_strategy= stopping_strategy,
        sequence_strategy= sequence_strategy,
        epsilon= epsilon,
        limit_n_variables= limit_n_variables,
        binning_indices= binning_indices,
        binning_strategy= binning_strategy,
        midway_binning= midway_binning
        )

    while(bp_class.stop_generating_sequences() == False):

        bp_class.generate_shapley_sequence()
        bp_class.reset_after_sequence()

        #print("Newly generated sequence: {}".format(bp_class.total_sequence))

        while(bp_class.early_sequence_stopping() == False):
            bp_class.next_variable_sequence()
            
            #print("Current sequence: {}".format(bp_class.current_sequence))

            bp_class.determine_shapley_value()
            bp_class.update_shapley_value()


    print('Average UD Shapley {}'.format(bp_class.average_shapley_values))
    print('Which sums up to: {}'.format(sum(bp_class.average_shapley_values.values())))
    
    bp_class.divide_average_shapley_by_Y()
    
    print("Average Dependency Shapley {}".format(bp_class.average_shapley_values))
    print('Which sums up to: {}'.format(sum(bp_class.average_shapley_values.values())))

    print('UD of all X_variables: {}'.format(bp_class.UD_all_X_Y))
    print('UD of Y, Y: {}'.format(bp_class.UD_Y_Y))
    print('Dependency of all X_variables: {}'.format(bp_class.UD_all_X_Y /bp_class.UD_Y_Y ))
    print(bp_class.n_sequences_counter)
    return(bp_class.average_shapley_values)

# %%
with cProfile.Profile() as pr:
    bp_feature_importance(dataset, X_indices, Y_indices, stopping_strategy = 24, sequence_strategy= 'exhaustive', epsilon= 0.0, limit_n_variables= 5, binning_indices= None, binning_strategy = 'auto', midway_binning = False)


stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
stats.dump_stats(filename='needs_profiling.prof')

# bp_feature_importance(dataset, X_indices, Y_indices, stopping_strategy = 120, sequence_strategy= 'random', epsilon= 0.0, limit_n_variables= 2, binning_indices= None, binning_strategy = 'auto', midway_binning = False)

# %%
# from tqdm.notebook import tqdm
# coefficients = [i for i in range(10)]
# test_results = []

# X_indices= [0,1]
# Y_indices = [2]
# binning_indices = None

# n_observations = 20000000
# # X_1 = np.random.uniform(low = 0.0, high= 1.0, size = n_observations)
# # X_2 = np.random.uniform(low = 0.0, high= 1.0, size = n_observations)

# X_1 = np.random.randint(2, size = n_observations)
# X_2 = np.random.randint(2, size = n_observations)

# for i, coeff in enumerate(tqdm(coefficients)):

#     Y = X_1 + coeff * X_2
#     dataset = np.stack((X_1, X_2, Y), axis = 1)

#     test_results.append(bp_feature_importance(dataset, X_indices, Y_indices, stopping_strategy = 150000, sequence_strategy= 'exhaustive', epsilon= 0.0, limit_n_variables= 2, binning_indices= binning_indices, binning_strategy = 'auto', midway_binning = False))
# # %%
# import matplotlib.pyplot as plt
# result_1 = [list(test_results[i].values())[0] for i in range(len(coefficients))]
# result_2 = [list(test_results[i].values())[1] for i in range(len(coefficients))]
# plt.plot(coefficients, result_1)
# plt.plot(coefficients, result_2, color = 'red')
# %%
