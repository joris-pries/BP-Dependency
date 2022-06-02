# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import math
from sklearn.metrics import mean_absolute_error
from collections import defaultdict


class dataset_results:
    def __init__(self, path_name, data_name) -> None:
        self.path_name = path_name
        self.data_name = data_name
        self.loss_function = mean_absolute_error
        self.load_result()
        self.remove_results()
        self.evaluate_result()
        self._grouped_result_dict()
        self._grouped_loss()

    def load_result(self):
        with open(self.path_name, 'rb') as f:
            # TODO X wordt niet opgeslagen, terwijl ik wel de shape gebruik. Hier moet ik nog iets mee
            if 'results_after' in self.path_name:
                self.result_dict, self.time_dict, self.did_not_work, self.not_finished_in_time, self.time_limit, self.test_methods = pickle.load(f)
                self.X = np.zeros(shape= (1,10))
            else:
                self.X, self.Y, self.labelencoded_Y, self.onehotencoded_Y, self.dataset, self.result_dict, self.time_dict, self.did_not_work, self.not_finished_in_time, self.time_limit, self.test_methods = pickle.load(f)

        return


    def remove_results(self, extra_method_names = []):
        for method in self.did_not_work + self.not_finished_in_time + extra_method_names:
            try:
                del self.result_dict[method]
                del self.time_dict[method]
                self.test_methods.remove(method)
            except:
                pass
        return


    def evaluate_result(self):
        # just a default value for y_true
        y_true = np.zeros(shape= self.X.shape[1])

        if self.data_name == 'decimal_system':
            y_true= np.repeat(1/3, 3)

        if self.data_name == 'binary_system':
            y_true= np.repeat(1/10, 10)

        self.loss = {}
        for method_name, pred in self.result_dict.items():
            if math.isnan(np.sum(pred)):
                self.loss[method_name] = np.NaN
            elif np.inf in pred:
                self.loss[method_name] = np.inf
            else:
                self.loss[method_name] = self.loss_function(y_true, y_pred= pred)


    def _grouped_result_dict(self):
        self.grouped_results = defaultdict(list)
        for key, value in self.result_dict.items():
            self.grouped_results[frozenset(value)].append(key)

    def _grouped_loss(self):
        self.grouped_loss = defaultdict(list)
        for key, value in self.loss.items():
            self.grouped_loss[value].append(key)

    # def _plot_loss(self):
    #     plt.plot(self.loss.keys(), self.loss.values())

# %%
# a = dataset_results('results/hiring_system_200-2022_05_27_21_18_22.pickle', '_')
# result_obtained = dataset_results('results/decimal_system_2000.pickle', 'decimal_system')
result_obtained = dataset_results('results/binary_system_200-2022_06_01_00_54_30.pickle', 'binary_system')

# result_obtained = dataset_results('results/binary_system_200/results_after_533.pickle', 'binary_system')

# a,b = load_result('results/decimal_system_200.pickle', ['result_dict', 'time_dict'])
# %%

pd_result_loss = pd.DataFrame(result_obtained.loss, index= [0]).T
# pd_result_loss.rename(columns= {0:'loss'})
pd_result_loss.set_axis(['loss'], axis= 1, inplace= True)
pd_result_loss.sort_values(by = 'loss', na_position= 'last', inplace= True)
# %%
pd_result_loss.head(100)
# %%
# # %%
# sorted(a.grouped_loss.items())
# # %%
# np.sort([i for i in a.grouped_loss.keys()])
# # %%
# sorted(a.grouped_loss)
# %%
