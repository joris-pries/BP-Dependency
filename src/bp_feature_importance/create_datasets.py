# %%
# Here, I create the datasets used in the experiments to check the properties
import enum
import numpy as np
from all_fi_measures import convert_to_labelencoded
import itertools
import pickle
from feature_importance import bp_feature_importance
import sys, os

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# %%
def save_function(X,Y, save_path):
    labelencoded_Y = convert_to_labelencoded(Y)
    # TODO Even dit uitgezet, want onehotencoded werd veels te groot door n_observations = 200000
    onehotencoded_Y = np.inf
    # onehotencoded_Y = convert_to_onehotencoded(Y)

    dataset = np.append(X, np.expand_dims(Y, axis = 1), axis = 1)

    with open(save_path + '.pickle', 'wb') as f:
        pickle.dump([X, Y, labelencoded_Y, onehotencoded_Y, dataset], f)

    try:
        X_indices = [i for i in range(X.shape[1])]
    except:
        # X is one-dimensional
        X_indices = [0]
    try:
        Y_indices = [len(X_indices) + i for i in range(Y.shape[1])]
    except:
        # Y is one-dimensional
        Y_indices = [len(X_indices)]


    with HiddenPrints():
        results = bp_feature_importance(dataset, X_indices, Y_indices)


    print(f'Results of {save_path}: {np.array([results[frozenset({i})] for i in range(X.shape[1])])}')


def sample_equally_from(X_options, Y_options, n_samples):
    if X_options.shape[0] ** 2 > n_samples:
        print('You probably need more samples to avoid uniqueness problems')

    if n_samples % X_options.shape[0] != 0:
        print('n_samples is not compatible')


    each_option_times = n_samples / X_options.shape[0]
    index_array = np.repeat(range(X_options.shape[0]), each_option_times)
    np.random.shuffle(index_array)

    try:
        X = np.array([X_options[i, :] for i in index_array])
    except:
        X = np.array([X_options[i] for i in index_array])

    try:
        Y = np.array([Y_options[i, :] for i in index_array])
    except:
        Y = np.array([Y_options[i] for i in index_array])

    return(X,Y)
# %%
########################################################
# Binary_system
save_path = 'datasets//experiment_10//binary_system'
n_samples = 1000
X_1_options = [0,1]
X_2_options = [0,1]
X_3_options = [0,1]

X_options = np.array(list(itertools.product(*[X_1_options, X_2_options, X_3_options])))

Y_options = np.zeros(X_options.shape[0], dtype= 'int_')
for i, [X_1, X_2, X_3] in enumerate(X_options):
    Y_options[i] = X_1 + 2 * X_2 + 4 * X_3

X, Y = sample_equally_from(X_options, Y_options, n_samples)

save_function(X, Y, save_path)

########################################################
# Binary_system_with_clone_X_1
save_path = 'datasets//experiment_10//binary_system_with_clone_X_1'
n_samples = 1000
X_1_options = [0,1]
X_2_options = [0,1]
X_3_options = [0,1]

X_options = np.array(list(itertools.product(*[X_1_options, X_2_options, X_3_options])))
# Cloning X_1 and adding it to the front
# X_1_clone, X_1, X_2, X_3
X_options = np.hstack((X_options[:, 0][:, None], X_options))

Y_options = np.zeros(X_options.shape[0], dtype= 'int_')
for i, [X_1_clone, X_1, X_2, X_3] in enumerate(X_options):
    Y_options[i] = X_1 + 2 * X_2 + 4 * X_3

X, Y = sample_equally_from(X_options, Y_options, n_samples)

save_function(X, Y, save_path)

########################################################
# Binary_system_with_clone_X_1_and_Y_squared
save_path = 'datasets//experiment_10//binary_system_with_clone_X_1_and_Y_squared'
n_samples = 1000
X_1_options = [0,1]
X_2_options = [0,1]
X_3_options = [0,1]

X_options = np.array(list(itertools.product(*[X_1_options, X_2_options, X_3_options])))
# Cloning X_1 and adding it to the front
# Features: X_1_clone, X_1, X_2, X_3
X_options = np.hstack((X_options[:, 0][:, None], X_options))


Y_options = np.zeros(X_options.shape[0], dtype= 'int_')
for i, [X_1_clone, X_1, X_2, X_3] in enumerate(X_options):
    Y_options[i] = X_1 + 2 * X_2 + 4 * X_3

# Adding Y squared to the end
# Features: X_1_clone, X_1, X_2, X_3, Y_squared
X_options = np.hstack((X_options, Y_options[:, None] ** 2))

X, Y = sample_equally_from(X_options, Y_options, n_samples)

save_function(X, Y, save_path)

########################################################
# Binary_system_with_clone_X_1_and_Y_squared_and_Y_cubed
save_path = 'datasets//experiment_10//binary_system_with_clone_X_1_and_Y_squared_and_Y_cubed'
n_samples = 1000
X_1_options = [0,1]
X_2_options = [0,1]
X_3_options = [0,1]

X_options = np.array(list(itertools.product(*[X_1_options, X_2_options, X_3_options])))
# Cloning X_1 and adding it to the front
# Features: X_1_clone, X_1, X_2, X_3
X_options = np.hstack((X_options[:, 0][:, None], X_options))


Y_options = np.zeros(X_options.shape[0], dtype= 'int_')
for i, [X_1_clone, X_1, X_2, X_3] in enumerate(X_options):
    Y_options[i] = X_1 + 2 * X_2 + 4 * X_3

# Adding Y squared and Y cubed to the end
# Features: X_1_clone, X_1, X_2, X_3, Y_squared, Y_cubed
X_options = np.hstack((X_options, Y_options[:, None] ** 2, Y_options[:, None] ** 3))

X, Y = sample_equally_from(X_options, Y_options, n_samples)

save_function(X, Y, save_path)

########################################################
# Binary_system_with_clone_X_1_and_Y_squared_and_Y_cubed_different_order
save_path = 'datasets//experiment_10//binary_system_with_clone_X_1_and_Y_squared_and_Y_cubed_different_order'
n_samples = 1000
X_1_options = [0,1]
X_2_options = [0,1]
X_3_options = [0,1]

X_options = np.array(list(itertools.product(*[X_1_options, X_2_options, X_3_options])))
# Cloning X_1 and adding it to the front
# Features: X_1_clone, X_1, X_2, X_3
X_options = np.hstack((X_options[:, 0][:, None], X_options))


Y_options = np.zeros(X_options.shape[0], dtype= 'int_')
for i, [X_1_clone, X_1, X_2, X_3] in enumerate(X_options):
    Y_options[i] = X_1 + 2 * X_2 + 4 * X_3

# Adding Y squared and Y cubed to the end
# Features: X_1_clone, X_1, X_2, X_3, Y_squared, Y_cubed
X_options = np.hstack((X_options, Y_options[:, None] ** 2, Y_options[:, None] ** 3))

# changing the order of X_options:
X_options = X_options[:, [3, 4, 5, 0, 1, 2]]

X, Y = sample_equally_from(X_options, Y_options, n_samples)

save_function(X, Y, save_path)
# %%

########################################################
# X_1_dependent
save_path = 'datasets//experiment_10//X_1_dependent'
n_samples = 1000
X_1_options = [0,1]
X_2_options = [0,1]
X_3_options = [0,1]

X_options = np.array(list(itertools.product(*[X_1_options, X_2_options, X_3_options])))

Y_options = np.zeros(X_options.shape[0], dtype= 'int_')
for i, [X_1, X_2, X_3] in enumerate(X_options):
    Y_options[i] = X_1

X, Y = sample_equally_from(X_options, Y_options, n_samples)

save_function(X, Y, save_path)

########################################################
# X_1_X_2_dependent
save_path = 'datasets//experiment_10//X_1_X_2_dependent'
n_samples = 1000
X_1_options = [1,2]
X_3_options = [1,2]

X_options = np.array(list(itertools.product(*[X_1_options, X_3_options])))
# Cloning X_1 and adding it to the front
# Features: X_1_clone, X_1, X_3
X_options = np.hstack((X_options[:, 0][:, None], X_options))
# Squaring X_1 gives
# Features: X_1, X_1_squared, X_3
X_options[:, 1] = X_options[:, 1] ** 2

Y_options = np.zeros(X_options.shape[0], dtype= 'int_')
for i, [X_1, X_2, X_3] in enumerate(X_options):
    Y_options[i] = X_1

X, Y = sample_equally_from(X_options, Y_options, n_samples)

save_function(X, Y, save_path)

########################################################
# X_1_X_2_X_3_dependent
save_path = 'datasets//experiment_10//X_1_X_2_X_3_dependent'
n_samples = 1000
X_1_options = [1,2]

X_options = np.array(list(itertools.product(*[X_1_options])))
# Cloning X_1 and adding it to the front (twice)
# Features: X_1_clone, X_1_clone X_1
X_options = np.hstack((X_options[:, 0][:, None], X_options[:, 0][:, None], X_options))
# Squaring and cubing gives
# Features: X_1, X_1_squared, X_1_cubed
X_options[:, 1] = X_options[:, 1] ** 2
X_options[:, 2] = X_options[:, 2] ** 3

Y_options = np.zeros(X_options.shape[0], dtype= 'int_')
for i, [X_1, X_2, X_3] in enumerate(X_options):
    Y_options[i] = X_1

X, Y = sample_equally_from(X_options, Y_options, n_samples)

save_function(X, Y, save_path)

# %%

########################################################
# Independent_system
save_path = 'datasets//experiment_10//independent_system'
n_samples = 2000
X_1_options = [0,1]
X_2_options = [0,1]
X_3_options = [0,1]

X_options = np.array(list(itertools.product(*[X_1_options, X_2_options, X_3_options])))

# Doubling X_options
X_options = np.vstack((X_options, X_options))

Y_options = np.array([0] * int(X_options.shape[0]/ 2) + [1] * int(X_options.shape[0] / 2))

X, Y = sample_equally_from(X_options, Y_options, n_samples)

save_function(X, Y, save_path)


########################################################
# Independent_system_with_constant_variable_X_4
save_path = 'datasets//experiment_10//independent_system_with_constant_variable_X_4'
n_samples = 2000
X_1_options = [0,1]
X_2_options = [0,1]
X_3_options = [0,1]
X_4_options = [1]


X_options = np.array(list(itertools.product(*[X_1_options, X_2_options, X_3_options, X_4_options])))

# Doubling X_options
X_options = np.vstack((X_options, X_options))

Y_options = np.array([0] * int(X_options.shape[0]/ 2) + [1] * int(X_options.shape[0] / 2))

X, Y = sample_equally_from(X_options, Y_options, n_samples)

save_function(X, Y, save_path)



# %%
########################################################
# uniform_system_more_bins
save_path = 'datasets//experiment_10//uniform_system_more_bins'
n_samples = 1000
Y = np.linspace(start= 0, stop = 1, num= n_samples, endpoint= True)

def round_bin_function(x, binspace):
    for i, value in enumerate(binspace):
        if x < value:
            if i == 0:
                return(-1)
            else:
                return(binspace[i-1])
    return(binspace[-1])

X_1_binspace = np.linspace(start= 0, stop = 1, num= int(n_samples / 100), endpoint= True)
X_2_binspace = np.linspace(start= 0, stop = 1, num= int(n_samples / 50), endpoint= True)
X_3_binspace = np.linspace(start= 0, stop = 1, num= int(n_samples / 20), endpoint= True)
X_4_binspace = np.linspace(start= 0, stop = 1, num= int(n_samples / 10), endpoint= True)
X_5_binspace = np.linspace(start= 0, stop = 1, num= int(n_samples / 1), endpoint= True)

X_1 = np.zeros(Y.shape)
X_2 = np.zeros(Y.shape)
X_3 = np.zeros(Y.shape)
X_4 = np.zeros(Y.shape)
X_5 = np.zeros(Y.shape)

for i in range(n_samples):
    X_1[i] = round_bin_function(Y[i], X_1_binspace)
    X_2[i] = round_bin_function(Y[i], X_2_binspace)
    X_3[i] = round_bin_function(Y[i], X_3_binspace)
    X_4[i] = round_bin_function(Y[i], X_4_binspace)
    X_5[i] = round_bin_function(Y[i], X_5_binspace)

X = np.vstack((X_1, X_2, X_3,X_4, X_5)).T

save_function(X, Y, save_path)


########################################################
# uniform_system_more_bins_3_variables
save_path = 'datasets//experiment_10//uniform_system_more_bins_3_variables'
n_samples = 1000
Y = np.linspace(start= 0, stop = 1, num= n_samples, endpoint= True)

def round_bin_function(x, binspace):
    for i, value in enumerate(binspace):
        if x < value:
            if i == 0:
                return(-1)
            else:
                return(binspace[i-1])
    return(binspace[-1])

X_1_binspace = np.linspace(start= 0, stop = 1, num= int(n_samples / 100), endpoint= True)
X_3_binspace = np.linspace(start= 0, stop = 1, num= int(n_samples / 20), endpoint= True)
X_5_binspace = np.linspace(start= 0, stop = 1, num= int(n_samples / 1), endpoint= True)

X_1 = np.zeros(Y.shape)
X_3 = np.zeros(Y.shape)
X_5 = np.zeros(Y.shape)

for i in range(n_samples):
    X_1[i] = round_bin_function(Y[i], X_1_binspace)
    X_3[i] = round_bin_function(Y[i], X_3_binspace)
    X_5[i] = round_bin_function(Y[i], X_5_binspace)

X = np.vstack((X_1, X_3, X_5)).T

save_function(X, Y, save_path)



########################################################
# uniform_system_more_bins_3_variables
save_path = 'datasets//experiment_10//uniform_system_more_bins_3_variables_different_order_and_clone'
n_samples = 1000
Y = np.linspace(start= 0, stop = 1, num= n_samples, endpoint= True)

def round_bin_function(x, binspace):
    for i, value in enumerate(binspace):
        if x < value:
            if i == 0:
                return(-1)
            else:
                return(binspace[i-1])
    return(binspace[-1])

X_1_binspace = np.linspace(start= 0, stop = 1, num= int(n_samples / 100), endpoint= True)
X_3_binspace = np.linspace(start= 0, stop = 1, num= int(n_samples / 20), endpoint= True)
X_5_binspace = np.linspace(start= 0, stop = 1, num= int(n_samples / 1), endpoint= True)

X_1 = np.zeros(Y.shape)
X_3 = np.zeros(Y.shape)
X_5 = np.zeros(Y.shape)


for i in range(n_samples):
    X_1[i] = round_bin_function(Y[i], X_1_binspace)
    X_3[i] = round_bin_function(Y[i], X_3_binspace)
    X_5[i] = round_bin_function(Y[i], X_5_binspace)

X_5_clone = X_5.copy()
#different order
X = np.vstack((X_5, X_3, X_1, X_5_clone)).T

save_function(X, Y, save_path)
# %%
########################################################
# XOR_dataset
save_path = 'datasets//experiment_10//XOR_dataset'
n_samples = 1000
X_1_options = [0,1]
X_2_options = [0,1]

X_options = np.array(list(itertools.product(*[X_1_options, X_2_options])))

Y_options = np.zeros(X_options.shape[0], dtype= 'int_')
for i, [X_1, X_2] in enumerate(X_options):
    Y_options[i] = X_1 * (1 - X_2) + X_2 * (1 - X_1)

X, Y = sample_equally_from(X_options, Y_options, n_samples)

save_function(X, Y, save_path)

########################################################
# XOR_dataset_with_null_independent
save_path = 'datasets//experiment_10//XOR_dataset_with_null_independent'
n_samples = 1000
X_1_options = [0,1]
X_2_options = [0,1]
X_3_options = [0,3]

X_options = np.array(list(itertools.product(*[X_1_options, X_2_options, X_3_options])))

Y_options = np.zeros(X_options.shape[0], dtype= 'int_')
for i, [X_1, X_2, X_3] in enumerate(X_options):
    Y_options[i] = X_1 * (1 - X_2) + X_2 * (1 - X_1)

X, Y = sample_equally_from(X_options, Y_options, n_samples)

save_function(X, Y, save_path)

########################################################
# XOR_dataset_with_clone
save_path = 'datasets//experiment_10//XOR_dataset_with_clone'
n_samples = 1000
X_1_options = [0,1]
X_2_options = [0,1]

X_options = np.array(list(itertools.product(*[X_1_options, X_2_options])))

Y_options = np.zeros(X_options.shape[0], dtype= 'int_')
for i, [X_1, X_2] in enumerate(X_options):
    Y_options[i] = X_1 * (1 - X_2) + X_2 * (1 - X_1)

X_options = np.hstack((X_options[:, 0][:, None], X_options))


X, Y = sample_equally_from(X_options, Y_options, n_samples)

save_function(X, Y, save_path)
# %%
########################################################
# Probability_dataset
def create_probability_dataset(prob):
    save_path = f'datasets//experiment_10//Probability_dataset_{str.replace(str(prob), ".", "_")}'
    n_samples = 1000


    X_1_options = [0,2]
    X_2_options = [0,2]


    X_options_p = np.array(list(itertools.product(*[X_1_options, X_2_options])))
    Y_options_p = np.zeros(X_options_p.shape[0], dtype= 'int_')
    for i, [X_1, X_2] in enumerate(X_options_p):
        Y_options_p[i] = X_1 // 2

    X_p, Y_p = sample_equally_from(X_options_p, Y_options_p, prob * n_samples)

    X_1_options = [1,3]
    X_2_options = [1,3]

    X_options_1_p = np.array(list(itertools.product(*[X_1_options, X_2_options])))
    Y_options_1_p = np.zeros(X_options_1_p.shape[0], dtype= 'int_')
    for i, [X_1, X_2] in enumerate(X_options_1_p):
        Y_options_1_p[i] = X_2 // 2

    X_1_p, Y_1_p = sample_equally_from(X_options_1_p, Y_options_1_p, n_samples  -prob * n_samples)

    if prob != 1 and prob != 0:
        X = np.vstack((X_p, X_1_p))
        Y = np.hstack((Y_p, Y_1_p))
    if prob == 1:
        X = X_p
        Y = Y_p
    if prob == 0:
        X = X_1_p
        Y = Y_1_p

    save_function(X, Y, save_path)


create_probability_dataset(0.0)
create_probability_dataset(0.1)
create_probability_dataset(0.2)
create_probability_dataset(0.3)
create_probability_dataset(0.4)
create_probability_dataset(0.5)
create_probability_dataset(0.6)
create_probability_dataset(0.7)
create_probability_dataset(0.8)
create_probability_dataset(0.9)
create_probability_dataset(1.0)
# %%
# ########################################################
# # Pairwise_combined_max_dataset
# save_path = 'datasets//experiment_10//Pairwise_combined_max_dataset'
# n_samples = 1000000
# X_1_options = [0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
# X_2_options = [0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,  15, 16, 17, 18, 19]
# X_3_options = [0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

# X_options = np.array(list(itertools.product(*[X_1_options, X_2_options, X_3_options])))

# Y_options = np.zeros(X_options.shape[0], dtype= 'int_')

# for i, [X_1, X_2, X_3] in enumerate(X_options):
#     Y_options[i] = np.max((X_1,X_2,X_3))

# Z_01 = np.max(X_options[:, [0,1]], axis = 1)
# Z_02 = np.max(X_options[:, [0,2]], axis = 1)
# Z_12 = np.max(X_options[:, [1,2]], axis = 1)

# X_options = np.hstack((X_options, Z_01[:, None], Z_02[:, None], Z_12[:, None]))


# X, Y = sample_equally_from(X_options, Y_options, n_samples)

# save_function(X, Y, save_path)

# %%
