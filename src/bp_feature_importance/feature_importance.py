# %%
from cmath import inf
from glob import glob
from operator import index
import numpy as np
import pandas as pd
import random
import itertools
import math
from bp_dependency import *


#__all__ = ['bin_data', 'unordered_bp_dependency', 'bp_dependency']

# %%
n_observations = 200
n_x_variables = 5

X = np.random.randint(2, size=(n_observations, n_x_variables))
Y_random_index = random.choices(range(n_x_variables), weights=[0.2, 0.1, 0.4, 0.1, 0.2], k= n_observations)
Y = np.asarray([X[i, Y_random_index[i]] for i in range(n_observations)]).reshape((n_observations, 1))

dataset = np.hstack((X,Y))
# %%

# Functie die input format
def format_input_variables(**kwargs):

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




# Functie die globals initialiseert
def init_globals(sequence_strategy, X_indices):

    global sequence_counter
    sequence_counter  = -1

    global average_shapley_values
    average_shapley_values = np.zeros(shape= X_indices)

    global average_shapley_values_counters
    average_shapley_values_counters = np.zeros(shape= X_indices)

    if sequence_strategy == 'exhaustive':        
        global sequence_array
        index_list = X_indices.copy()
        sequence_array = np.asarray(list(itertools.permutations(index_list)))


# Functie die globals() opschoont
def clear_globals():
    copy_globals = globals().copy()
    for global_variable in copy_globals:
        if global_variable in ['sequence_array', 'sequence_counter']:
            del globals()[global_variable]

    del copy_globals

# Functie die bepaald of er gestopt moet worden met het hele proces
def stop_generating_sequences(stopping_strategy):

    global sequence_counter

    if isinstance(stopping_strategy, int):
        # In this case, we want to generate exactly stopping_strategy sequences (that is why -1)
        return(sequence_counter >= stopping_strategy - 1)





# Functie die bepaald of er gestopt moet worden met 1 sequence
def early_sequence_stopping(shapley_values_current_sequence, epsilon= 0.0, n_variables= np.inf):
    total_sum = np.sum(shapley_values_current_sequence)
    if 1.0 - total_sum < epsilon:
        return(True)
    if len(shapley_values_current_sequence) >= n_variables:
        return(True)
    
    return(False)



# Functie die gemiddelde shapley values bijhoudt en bijvoorbeeld hoe vaak ze gekozen zijn om de gemiddeldes uit te rekenen
def updating_shapley_values(additional_shapley_values):
    global average_shapley_values
    global average_shapley_values_counters

    for i, value in additional_shapley_values:
        if not math.isnan(value):
            average_shapley_values[i] = (average_shapley_values_counters[i] * average_shapley_values[i] + value) / (average_shapley_values_counters[i] + 1)
            average_shapley_values_counters[i] += 1

    return
  
 

#Functie die shapley sequences genereert
def shapley_sequence(sequence_strategy, X_indices):
    global sequence_counter
    sequence_counter += 1

    index_list = X_indices.copy()

    # Gewoon willekeurig
    if sequence_strategy == 'random':
        random.shuffle(index_list)
        return(index_list)

    # Gebruikt een lijst om die 1 voor 1 af te lopen
    if sequence_strategy == 'exhaustive':
        global sequence_array
        return_sequence = sequence_array[sequence_counter, :]
        return(return_sequence)


#TODO: Functie die additional shapley values bepaald

#TODO: Functie die shaples values blijft genereren tot een stop conditie
def bp_feature_importance(dataset, X_indices, Y_indices, stopping_strategy = 24, sequence_strategy= 'exhaustive', epsilon= 0.0, n_variables= np.inf, binning_indices= None, binning_strategy = 'auto', midway_binning = False, format_input = True):

    dataset, X_indices, Y_indices, stopping_strategy, sequence_strategy, epsilon, n_variables, binning_indices, binning_strategy, midway_binning, format_input = format_input_variables(dataset= dataset, X_indices= X_indices, Y_indices= Y_indices, stopping_strategy= stopping_strategy, sequence_strategy= sequence_strategy, epsilon= epsilon, n_variables= n_variables, binning_indices= binning_indices, binning_strategy= binning_strategy, midway_binning= midway_binning, format_input= format_input)

    init_globals(sequence_strategy= sequence_strategy, X_indices= X_indices)

    while not stop_generating_sequences(stopping_strategy= stopping_strategy):
        shap_sequence = shapley_sequence(sequence_strategy= sequence_strategy, X_indices= X_indices)
        
        print(sequence_counter)

    clear_globals()
    print("test")
# %%

# %%
bp_feature_importance(dataset= np.zeros(shape = (6,6)), X_indices= [4,5,6,7], Y_indices=[1], stopping_strategy= 24)


# %%
