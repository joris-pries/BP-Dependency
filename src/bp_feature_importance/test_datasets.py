# %%
from operator import xor
from matplotlib import pyplot
import numpy as np
import random

from torch import combinations
# %%
# # n_samples = 2000000
# # n_help = 100
# # # X_1 = np.random.uniform(low = 0, high= 1, size = n_samples)
# # # X_2 = np.random.uniform(low = 0, high= 2, size = n_samples)
# # # X_3 = np.random.uniform(low = 0, high= 3, size = n_samples)
# # # X_1 = np.random.randint(low = - 1 * n_help, high= 1 * n_help, size = n_samples)
# # # X_2 = np.random.randint(low = - 2 * n_help, high= 2 * n_help, size = n_samples)
# # # X_3 = np.random.randint(low = - 3 * n_help, high= 3 * n_help, size = n_samples)
# # # X_1 = np.random.randint(low = 0, high= 1 * n_help, size = n_samples)
# # # X_2 = np.random.randint(low = 0, high= 2 * n_help, size = n_samples)
# # # X_3 = np.random.randint(low = 0, high= 3 * n_help, size = n_samples)
# # # # X_4 = np.random.uniform(low = 0, high= 4, size = n_samples)
# # # # X_1 = np.random.randint(low = 0, high= 2, size = n_samples)
# # # # X_2 = np.random.randint(low = 0, high= 3, size = n_samples)
# # # # X_3 = np.random.randint(low = 0, high= 4, size = n_samples)
# # # X = np.stack((X_1,X_2,X_3), axis =1)
# # z = 1/6
# # q = 1/3

# # X_1 = np.random.choice([0,1], size= n_samples, replace= True)
# # X_2 = np.random.choice([0,1], size= n_samples, replace= True)
# # # X_3 = np.array([np.random.choice([X_1[i], X_2[i]], p= [q, 1-q]) for i in range(n_samples)])
# # X_3 = np.array([np.random.choice([X_1[i], X_2[i]], p= [q, 1-q]) for i in range(n_samples)])


# # # X_4 = np.random.choice([0,1], size= n_samples, replace= True)
# # # X_5 = np.random.choice([0,1], size= n_samples, replace= True)
# # # X_6 = np.random.choice([0,1], size= n_samples, replace= True)
# # # X_7 = np.random.choice([0,1], size= n_samples, replace= True)


# # # X = np.stack((X_1,X_2,X_3), axis =1)
# # # X = np.stack((X_1,X_2,X_3,X_4,X_5), axis =1)
# # # X = np.stack((X_1,X_2,X_3,X_4,X_5,X_6,X_7), axis =1)

# # S = np.random.choice([0,1], p = [z, 1-z], replace= True, size= n_samples)
# # Y = np.array([[X_1[i], X_2[i]][S[i]] for i in range(n_samples)])

# # X = np.stack((X_1,X_2,X_3,S), axis =1)

# # # Y = [np.sum(X[i,:])>= 4 for i in range(X.shape[0])]

# # # Y = np.min(X, axis = 1)




# # # X_1 = np.random.randint(low = 0, high = n_help, size = n_samples)
# # # X_2 = np.random.randint(low = 0, high = n_help, size = n_samples)
# # # X_3 = np.random.randint(low = 0, high = n_help, size = n_samples)

# # # print(np.array_equal(X_1, X_1_old))
# # # print(np.array_equal(X_2, X_2_old))
# # # print(np.array_equal(X_3, X_3_old))



# # # Z_12 = np.max(np.stack((X_1, X_2), axis = 1), axis = 1)
# # # Z_13 = np.max(np.stack((X_1, X_3), axis = 1), axis = 1)
# # # Z_23 = np.max(np.stack((X_2, X_3), axis = 1), axis = 1)

# # # X = np.stack((X_1, X_2, X_3, Z_12, Z_13, Z_23), axis =1)
# # # X = np.stack((X_1, X_2, X_3), axis = 1)

# # # Y = [max(X_1[i] * X_2[i], X_2[i] * X_3[i], X_3[i] * X_4[i]) for i in range(n_samples)]

# # # Y = np.max(np.stack((X_1, X_2, X_3), axis = 1), axis = 1)

# # # print(np.mean(Z_12 == Y))
# # # print(np.mean(Z_13 == Y))
# # # print(np.mean(Z_23 == Y))

# # # Y = np.max(X, axis = 1)

# # # dataset = np.stack((X_1,X_2,X_3,Y), axis =1)
# # dataset = np.append(X, np.expand_dims(Y, axis = 1), axis = 1)


# # try:
# #     X_indices = [i for i in range(X.shape[1])]
# # except:
# #     # X is one-dimensional
# #     X_indices = [0]
# # try:
# #     Y_indices = [len(X_indices) + i for i in range(Y.shape[1])]
# # except:
# #     # Y is one-dimensional
# #     Y_indices = [len(X_indices)]


# # X_indices = [[0,3], [1,3],[2,3]]
# # Y_indices = [-1]

# # # argmaxY = np.argmax(X, axis = 1)
# # # [np.sum(argmaxY == i)/ n_samples for i in range(X.shape[1])]
# # # %%
# # # print([1/96, 31/288, 91/288, 163/288])
# # # %%
# # import cProfile
# # import pstats
# # import time
# # from feature_importance import bp_feature_importance
# # start = time.perf_counter()
# # with cProfile.Profile() as pr:
# #     results = bp_feature_importance(dataset, X_indices, Y_indices, sequence_strategy= 'exhaustive', binning_indices= None)
# #     # results = bp_feature_importance(dataset, X_indices, Y_indices, sequence_strategy= 'exhaustive', binning_indices= range(dataset.shape[1]))
# # end = time.perf_counter()
# # print(f'Finished in {round(end-start,2 )} second(s)')

# # stats = pstats.Stats(pr)
# # stats.sort_stats(pstats.SortKey.TIME)
# # stats.dump_stats(filename='needs_profiling.prof')



# # # fi_results = np.array([results[frozenset({i})] for i in range(len(X_indices))])
# # # %%
# # pyplot.plot(fi_results)
# # # %%
# # [1/18, 11/36, 23/36]
# # # # %%
# # # [6/11, 3/11, 2/11]
# # # # %%

# # # def help_function(x):
# # #     for i in range(c.shape[0]):
# # #         if np.array_equal(x, c[i, :4]):
# # #             return c[i, 4]
# # #     return None
# # # # %%
# # # help_function(np.array([0,1,1,1]))
# # # # %%
# # # for i in range(c.shape[0]):
# # #     first_value = c[i,4]
# # #     old_array = c[i,:4]
# # #     print('\n')
# # #     print(old_array)
# # #     sum = 0
# # #     for j in range(c.shape[1] - 1):
# # #         new_array = old_array.copy()
# # #         new_array[j] = 1 - new_array[j]
# # #         new_value = help_function(new_array)
# # #         if new_value != first_value:
# # #             #print('x', end = '')
# # #             sum += 1
# # #         else:
# # #             pass
# # #             #print('o', end = '')
# # #     if sum == 0:
# # #         print(0)
# # #     else:
# # #         print(1 / sum)
# # # %%
# # # import itertools
# # # a = [[0,4],[0,2],[0,2], [0,1], [0,1], [0,1], [0,1]]
# # # b = list(itertools.product(*a))
# # # c = np.array([i for i in b if sum(i) >= 6])
# # # print(np.mean(c[:,0] == 4))
# # # print(np.mean(c[:,1] == 2))
# # # print(np.mean(c[:,-1] == 1))


# # # %%
# # fi_results[1] / fi_results[2]
# # # %%
# # q*fi_results[0] + (1-q) * fi_results[1]
# # # %%
# # fi_results[2]
# # # %%
# # np.mean(X_1 == Y) * q + np.mean(X_2 == Y) * (1-q)
# # # %%
# # np.mean(X_1 == Y)
# # # %%
# # np.mean(X_2 == Y)
# # %%
# import pickle
# def load_dataset(data_path):
#     # This will load the dataset in a certain path

#     with open(data_path, 'rb') as f:
#         X, Y, labelencoded_Y, onehotencoded_Y, dataset = pickle.load(f)

#     if X.ndim == 1:
#         X = np.expand_dims(X, axis=1)

#     return X, Y, labelencoded_Y, onehotencoded_Y, dataset

# X, Y, labelencoded_Y, onehotencoded_Y, dataset = load_dataset('datasets/experiment_6/prob_selected_033_2000.pickle')
# # %%
# result = determine_fi(fi_method_name='AdaBoost_Classifier', data_path='datasets/experiment_6/prob_selected_033_2000.pickle')
# # %%
# np.nan in X
# # %%
# for name in [
# 'min_function_expanding_dist_4_2000.pickle',
# 'min_function_expanding_dist_5_2000.pickle',
# 'min_function_increasing_dist_3_2000.pickle',
# 'min_function_increasing_dist_4_2000.pickle',
# 'min_function_increasing_dist_5_2000.pickle',
# 'pairwise_combined_max_2000.pickle',
# 'pairwise_combined_min.pickle',
# 'prob_selected_02_2000.pickle',
# 'prob_selected_025_2000.pickle',
# 'prob_selected_033_2000.pickle',
# 'binary_system_2000.pickle',
# 'cloned_decimal_system_2000.pickle',
# 'decimal_system_2000.pickle',
# 'decimal_system_with_independence_2000.pickle',
# 'independent_easy_2000.pickle',
# 'independent_hard_2000.pickle',
# 'max_function_equal_dist_2000.pickle',
# 'max_function_expanding_dist_3_2000.pickle',
# 'max_function_expanding_dist_4_2000.pickle',
# 'max_function_expanding_dist_5_2000.pickle',
# 'max_function_increasing_dist_3_2000.pickle',
# 'max_function_increasing_dist_4_2000.pickle',
# 'max_function_increasing_dist_5_2000.pickle',
# 'min_function_equal_dist_2000.pickle',
# 'min_function_expanding_dist_3_2000.pickle'
# ]:
#     X, Y, labelencoded_Y, onehotencoded_Y, dataset = load_dataset(f'datasets/experiment_6/{name}')
#     if None in Y:
#         print(name)
# # %%
# w = 0.25
# X_1 = np.random.randint(2, size=2000)
# X_2 = np.random.randint(2, size=2000)
# S =  np.random.choice([0,1], size=2000, replace= True, p= [w, 1-w])

# Y = np.array([[X_1[i], X_2[i]][S[i]] for i in range(2000)])

# def help_function(x, s):
#     if x ==0 & s == 0:
#         return 0
#     if x == 0 & s == 1:
#         return 1
#     if x == 1 & s == 0:
#         return 2
#     if x == 1 & s == 1:
#         return 3
# X_1_with_S = np.array([help_function(X_1[i], S[i]) for i in range(2000)])
# X_2_with_S = np.array([help_function(X_2[i], S[i]) for i in range(2000)])

# X = np.stack((X_1_with_S, X_2_with_S), axis=1)
# dataset = np.append(X, np.expand_dims(Y, axis = 1), axis = 1)
# %%
from feature_importance import bp_feature_importance
kwargs = {}
kwargs['n_observations'] = 20000
# K = 10
# X_1 = np.random.randint(K, size=kwargs['n_observations'])
# X_2 = np.random.randint(2 * K, size=kwargs['n_observations'])
# X_3 = np.random.randint(3 * K, size=kwargs['n_observations'])
# # X_4 = np.random.randint(4 * K, size=kwargs['n_observations'])

# X = np.stack((X_1, X_2, X_3), axis=1)
# Y = np.max(X, axis = 1)
# dataset = np.append(X, np.expand_dims(Y, axis = 1), axis = 1)

# X_1 = np.random.randint((-1 * K) + 1, K, size=kwargs['n_observations'])
# X_2 = np.random.randint((-2 * K) + 1, 2 * K, size=kwargs['n_observations'])
# X_3 = np.random.randint((-3 * K) + 1, 3 * K, size=kwargs['n_observations'])

# X = np.stack((X_1, X_2, X_3), axis=1)
# Y = np.max(X, axis = 1)
# dataset = np.append(X, np.expand_dims(Y, axis = 1), axis = 1)



# # %%
# K = 10
# X_1 = np.random.randint(K, size=kwargs['n_observations'])
# X_2 = np.random.randint(2 * K, size=kwargs['n_observations'])
# X_3 = np.random.randint(3 * K, size=kwargs['n_observations'])

# X = np.stack((X_1, X_2, X_3), axis=1)
# Y = np.min(X, axis = 1)
# dataset = np.append(X, np.expand_dims(Y, axis = 1), axis = 1)

# try:
#     X_indices = [i for i in range(X.shape[1])]
# except:
#     # X is one-dimensional
#     X_indices = [0]
# try:
#     Y_indices = [len(X_indices) + i for i in range(Y.shape[1])]
# except:
#     # Y is one-dimensional
#     Y_indices = [len(X_indices)]
# results = bp_feature_importance(dataset, X_indices, Y_indices, kwargs)
# fi_results = np.array([results[frozenset({i})] for i in range(X.shape[1])])
# # %%
# np.mean(Y <= 2)
# # %%
# import itertools
# # %%
# k_results = []
# for K in range(2, 21):
# # K = 2
# # X_1 = [i for i in range(-1 * K + (K != 1), 1 * K + (K == 1) )]
# # X_2 = [i for i in range(-2 * K + (K != 1), 2 * K + (K == 1))]
# # X_3 = [i for i in range(-3 * K + (K != 1), 3 * K + (K == 1))]
# # X_1 = [i for i in range(0, 1 * K + 1 * (K == 1))]
# # X_2 = [i for i in range(0, 2 * K  + 1 * (K == 1))]
# # X_3 = [i for i in range(0, 3 * K + 1 * (K == 1))]

#     X_1 = [i for i in range(0, 1 * K)]
#     X_2 = [i for i in range(0, 2 * K)]
#     X_3 = [i for i in range(0, 3 * K)]
#     X_4 = [i for i in range(0, 4 * K)]
#     # X_1 = [i for i in range(-1 * K + 1, 1 * K)]
#     # X_2 = [i for i in range(-2 * K + 1, 2 * K)]
#     # X_3 = [i for i in range(-3 * K + 1, 3 * K)]
#     # X_4 = [i for i in range(-4 * K + 1, 4 * K)]


#     combinations_list = list(itertools.product(X_1, X_2, X_3, X_4))
#     # print(combinations_list)
#     X = np.array(combinations_list)
#     # random_rows = np.random.choice(X.shape[0], size = 2000, replace= True)
#     # X = X[random_rows, :]
#     Y = np.min(X, axis = 1)
#     dataset = np.append(X, np.expand_dims(Y, axis = 1), axis = 1)
#     try:
#         X_indices = [i for i in range(X.shape[1])]
#     except:
#         # X is one-dimensional
#         X_indices = [0]
#     try:
#         Y_indices = [len(X_indices) + i for i in range(Y.shape[1])]
#     except:
#         # Y is one-dimensional
#         Y_indices = [len(X_indices)]
#     results = bp_feature_importance(dataset, X_indices, Y_indices, kwargs, sequence_strategy= 'exhaustive')
#     fi_results = np.array([results[frozenset({i})] for i in range(X.shape[1])])
#     k_results.append(fi_results)
# # %%
# k_results
# %%
# K = 10
# I_1 = np.random.randint(K, size=kwargs['n_observations'])
# I_2 = np.random.randint(K, size=kwargs['n_observations'])
# I_3 = np.random.randint(K, size=kwargs['n_observations'])
# I_4 = np.random.randint(K, size=kwargs['n_observations'])
# I_5 = np.random.randint(K, size=kwargs['n_observations'])

# Y = np.random.randint(K, size=kwargs['n_observations'])
# X = np.stack((I_1, I_2, I_3,I_4,I_5), axis=1)
# dataset = np.append(X, np.expand_dims(Y, axis = 1), axis = 1)

# try:
#     X_indices = [i for i in range(X.shape[1])]
# except:
#     # X is one-dimensional
#     X_indices = [0]
# try:
#     Y_indices = [len(X_indices) + i for i in range(Y.shape[1])]
# except:
#     # Y is one-dimensional
#     Y_indices = [len(X_indices)]
# results = bp_feature_importance(dataset, X_indices, Y_indices, kwargs)
# fi_results = np.array([results[frozenset({i})] for i in range(X.shape[1])])
# # %%
# fi_results
# %%

X_1 = np.random.randint(2, size=kwargs['n_observations'])
X_2 = np.random.randint(2, size=kwargs['n_observations'])
# X_2 =1 - np.zeros(kwargs['n_observations'])
Y = [0] * len(X_1)
for i in range(len(X_1)):
    leeftijd = X_1[i]
    geslacht = X_2[i]

    if geslacht == 0:
        Y[i] = np.random.choice([0,1], size = 1, p = [1/4, 3/4])[0]

    if geslacht == 1 and leeftijd == 0:
        Y[i] = np.random.choice([0,1], size = 1, p = [5/8, 3/8])[0]

    if geslacht == 1 and leeftijd == 1:
        Y[i] = np.random.choice([0,1], size = 1, p = [7/8, 1/8])[0]

X = np.stack((X_1, X_2), axis=1)

Y = np.array(Y)
dataset = np.append(X, np.expand_dims(Y, axis = 1), axis = 1)

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
results = bp_feature_importance(dataset, X_indices, Y_indices, kwargs)
fi_results = np.array([results[frozenset({i})] for i in range(X.shape[1])])
# %%
fi_results
# %%

sum_correct = 0
Y_pred = [0] * len(X_1)
for i in range(len(X_1)):
    leeftijd = X_1[i]
    geslacht = X_2[i]
    if leeftijd == 0:
        Y_pred[i] = np.random.choice([0,1], size = 1, p = [7/16, 9/16])[0]
    if leeftijd == 1:
        Y_pred[i] = np.random.choice([0,1], size = 1, p = [9/16, 7/16])[0]

    if Y_pred[i] == Y[i]:
        sum_correct += 1
sum_correct /= kwargs['n_observations']
print(sum_correct)
# %%
sum_correct = 0
Y_pred = [0] * len(X_1)
for i in range(len(X_1)):
    leeftijd = X_1[i]
    geslacht = X_2[i]
    if geslacht == 0:
        Y_pred[i] = np.random.choice([0,1], size = 1, p = [1/4, 3/4])[0]
    if geslacht == 1:
        Y_pred[i] = np.random.choice([0,1], size = 1, p = [3/4, 1/4])[0]

    if Y_pred[i] == Y[i]:
        sum_correct += 1
sum_correct /= kwargs['n_observations']
print(sum_correct)
# %%

X_1 = np.random.randint(2, size=kwargs['n_observations'])
X_2 = np.random.randint(2, size=kwargs['n_observations'])
Y = X_1 * (1 - X_2) + X_2 * (1 - X_1)

X = np.stack((X_1, X_2), axis=1)
dataset = np.append(X, np.expand_dims(Y, axis = 1), axis = 1)

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
results = bp_feature_importance(dataset, X_indices, Y_indices, kwargs)
fi_results = np.array([results[frozenset({i})] for i in range(X.shape[1])])
# %%
fi_results
# %%
w = 0.75

X_1 = np.random.randint(2, size=kwargs['n_observations'])
X_2 = np.random.randint(2, size=kwargs['n_observations'])
S =  np.random.choice([0,1], size=kwargs['n_observations'], replace= True, p= [w, 1-w])

Y = np.array([[X_1[i], X_2[i]][S[i]] for i in range(kwargs['n_observations'])])

def help_function(x, s):
    if x ==0 and s == 0:
        return 0
    if x == 0 and s == 1:
        return 1
    if x == 1 and s == 0:
        return 2
    if x == 1 and s == 1:
        return 3
X_1_with_S = np.array([help_function(X_1[i], S[i]) for i in range(kwargs['n_observations'])])
X_2_with_S = np.array([help_function(X_2[i], S[i]) for i in range(kwargs['n_observations'])])

X = np.stack((X_1_with_S, X_2_with_S), axis=1)
# X = np.stack((X_1, X_2), axis=1)

dataset = np.append(X, np.expand_dims(Y, axis = 1), axis = 1)

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
results = bp_feature_importance(dataset, X_indices, Y_indices, kwargs)
fi_results = np.array([results[frozenset({i})] for i in range(X.shape[1])])
# %%
fi_results
# %%
X_1 = np.random.randint(2, size=kwargs['n_observations'])
X_2 = np.random.randint(2, size=kwargs['n_observations'])

S_1 = np.random.choice([0,1], size = kwargs['n_observations'], p = [1/4, 3/4])
S_2 = np.random.choice([0,1], size = kwargs['n_observations'], p = [5/8, 3/8])
S_3 = np.random.choice([0,1], size = kwargs['n_observations'], p = [7/8, 1/8])

def convert_binary_to_integer(x):
    res = 0
    for ele in x:
        res = (res << 1) | ele
    return(res)

X_1_with_S_1 = np.array([convert_binary_to_integer(np.array([X_1[i], S_1[i]])) for i in range(kwargs['n_observations'])])
X_2_with_S_2_and_S_3 = np.array([convert_binary_to_integer(np.array([X_2[i], S_2[i], S_3[i]])) for i in range(kwargs['n_observations'])])
# %%

Y = [0] * len(X_1)
for i in range(len(X_1)):
    leeftijd = X_1[i]
    geslacht = X_2[i]

    if geslacht == 0:
        Y[i] = S_1[i]

    if geslacht == 1 and leeftijd == 0:
        Y[i] = S_2[i]

    if geslacht == 1 and leeftijd == 1:
        Y[i] = S_3[i]

# X = np.stack((X_1, X_2), axis=1)
X = np.stack((X_1_with_S_1, X_2_with_S_2_and_S_3), axis=1)

Y = np.array(Y)
dataset = np.append(X, np.expand_dims(Y, axis = 1), axis = 1)

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
results = bp_feature_importance(dataset, X_indices, Y_indices, kwargs)
fi_results = np.array([results[frozenset({i})] for i in range(X.shape[1])])
# %%
fi_results