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
# path_names = glob.glob('results/experiment_7_individual/cloned_decimal_system_2000/*.pickle')
# target_path = 'results/experiment_7_individual/cloned_decimal_system_2000/grouped_results/grouped'
# results_after_path = 'results/experiment_7/binary_system_2000/results_after_52.pickle'
# group_together_results(target_path, path_names, results_after_path)
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
        self.evaluate_result()
        self._grouped_result_dict()
        self._grouped_loss()

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
        # if 'Fisher_Score' in self.result_dict:
        #     if np.array_equal(self.result_dict['Fisher_Score'], np.array([np.inf] * self.X.shape[1])):
        #         self.did_not_work += ['Fisher_Score']

        for method in self.did_not_work + self.not_finished_in_time + extra_method_names:
        # for method in extra_method_names:
            try:
                del self.result_dict[method]
                del self.time_dict[method]
                # self.test_methods.remove(method)
            except:
                pass
        return


    def evaluate_result(self):
        # just a default value for y_true
        self.n_variables = self.X.shape[1]
        y_true = np.zeros(shape= self.n_variables)

        if 'independent_easy' in self.data_name:
            y_true= np.repeat(0, self.n_variables)

        if 'independent_hard' in self.data_name:
            y_true= np.repeat(1/self.n_variables, self.n_variables)

        if 'decimal_system' in self.data_name:
            y_true= np.repeat(1/self.n_variables, self.n_variables)

        if 'cloned_decimal_system' in self.data_name:
            y_true= np.repeat(1/self.n_variables, self.n_variables)

        if 'decimal_system_with_independence' in self.data_name:
            y_true= np.concatenate((np.repeat(1/3, 3), np.repeat(0, 5)))

        if 'binary_system' in self.data_name:
            y_true= np.repeat(1/self.n_variables, self.n_variables)

        if 'prob_selected_025_' in self.data_name:
            p = 1/4
            y_true = np.array([p, 1-p])

        if 'prob_selected_02_' in self.data_name:
            p = 1/5
            y_true = np.array([p, 1-p])

        if 'prob_selected_033_' in self.data_name:
            p = 1/3
            y_true = np.array([p, 1-p])

        if 'prob_selected_05_' in self.data_name:
            p = 1/2
            y_true = np.array([p, 1-p])

        if 'max_function_equal_dist' in self.data_name:
            y_true= np.repeat(1/self.n_variables, self.n_variables)

        if 'min_function_equal_dist' in self.data_name:
            y_true= np.repeat(1/self.n_variables, self.n_variables)

        if 'max_function_increasing_dist_3' in self.data_name:
            y_true = np.array([1/18, 11/36, 23/36])

        if 'max_function_increasing_dist_4' in self.data_name:
            y_true = np.array([1/96, 31/288, 91/288, 163/288])

        if 'max_function_increasing_dist_5' in self.data_name:
            y_true = np.array([1/600, 79/2400, 997/7200, 2257/7200, 3697/7200])


        if 'max_function_expanding_dist_3' in self.data_name:
            y_true = np.array([38/144, 47/144, 59/144])

        # if 'max_function_expanding_dist_4' in self.data_name:
        #     y_true = np.array([1/96, 31/288, 91/288, 163/288])

        # if 'max_function_expanding_dist_5' in self.data_name:
        #     y_true = np.array([1/600, 79/2400, 997/7200, 2257/7200, 3697/7200])


        if 'pairwise_combined_max' in self.data_name:
            y_true= np.concatenate((np.repeat(1/9, 3), np.repeat(2/9, 3)))

        if 'pairwise_combined_min' in self.data_name:
            y_true= np.concatenate((np.repeat(1/9, 3), np.repeat(2/9, 3)))



        # loss
        self.loss = {}
        for method_name, pred in self.result_dict.items():
            if math.isnan(np.sum(pred)):
                self.loss[method_name] = np.NaN
            elif np.inf in pred:
                self.loss[method_name] = np.inf
            else:
                self.loss[method_name] = self.loss_function(y_true, y_pred= pred)

            if method_name == 'bp_feature_importance':
                with open('bp_results_text.txt', 'a') as f:
                    f.write(str(self.data_name))
                    f.write(f'\ny_pred is: {pred}\n')
                    f.write(f'y_true is {y_true}\n\n')


        # sorted loss
        pd_result_loss = pd.DataFrame(self.loss, index= [0]).T
        pd_result_loss.set_axis(['loss'], axis= 1, inplace= True)
        pd_result_loss.sort_values(by = 'loss', na_position= 'last', inplace= True)
        self.sorted_loss = pd_result_loss['loss'].to_dict()

        # scaled loss
        self.scaled_loss = {}
        for method_name, pred in self.result_dict.items():

            if math.isnan(np.sum(pred)):
                self.scaled_loss[method_name] = np.NaN
                continue
            if np.inf in pred or -np.inf in pred:
                self.scaled_loss[method_name] = np.NaN
                continue

            if np.array_equal(pred, np.zeros(shape= self.X.shape[1])):
                pred = np.array([1 / self.X.shape[1]] * self.X.shape[1])
            else:
                pred /= np.sum(np.abs(pred))

            self.scaled_loss[method_name] = self.loss_function(y_true, y_pred= pred)

        # sorted scaled loss
        pd_result_scaled_loss = pd.DataFrame(self.scaled_loss, index= [0]).T
        pd_result_scaled_loss.set_axis(['loss'], axis= 1, inplace= True)
        pd_result_scaled_loss.sort_values(by = 'loss', na_position= 'last', inplace= True)
        self.sorted_scaled_loss = pd_result_scaled_loss['loss'].to_dict()

        # borda count
        self.borda_count = {}
        borda_count_help = len(self.test_methods)
        # + len(set(self.did_not_work + self.not_finished_in_time))
        for name in self.sorted_loss.keys():
            self.borda_count[name] = borda_count_help
            borda_count_help -= 1

        # did not work loss and not finished in time loss
        self.did_not_work_loss = {}
        self.not_finished_in_time_loss = {}
        for name in self.did_not_work:
            self.did_not_work_loss[name] = 1
        for name in self.not_finished_in_time:
            self.not_finished_in_time_loss[name] = 1
        # assuming that results are removed from did_not_work and not_finished_in_time
        for name in self.result_dict.keys():
            self.did_not_work_loss[name] = 0
            self.not_finished_in_time_loss[name] = 0






    def _grouped_result_dict(self):
        self.grouped_results = defaultdict(list)
        for key, value in self.result_dict.items():
            self.grouped_results[frozenset(np.squeeze(value))].append(key)

    def _grouped_loss(self):
        self.grouped_loss = defaultdict(list)
        for key, value in self.loss.items():
            self.grouped_loss[value].append(key)



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

        self.average_loss()
        self.sorted_average_loss()
        self._create_latex_table(table_type = 'average_loss', title = 'title')
        self._create_latex_table(table_type = 'top_k_sorted_average_loss', title = 'title')



    def average_loss(self):
        self.combined_loss = pd.DataFrame()
        for dataset_results_class in self.results_obtained.values():
            self.combined_loss = pd.concat([self.combined_loss, pd.DataFrame.from_records([dataset_results_class.loss])], ignore_index= True)
        # self.combined_loss = pd.DataFrame(loss_help).T
        self.combined_loss.set_axis(self.results_obtained.keys(), axis = 0, inplace = True)

        self.average_loss = self.combined_loss.mean(axis = 0)

    def sorted_average_loss(self):
        self.sorted_average_loss = self.average_loss.sort_values(na_position= 'last')


    # ! This does nothing yet
    def _create_latex_table(self, table_type, caption = '', label = '', title = ''):

        # some default settings
        table_string = ''
        cell_color = 'white' # ! at least for now
        n_columns = 4

        # ! open table
        table_string += r'\begin{table*}[]' + '\n' + r'\Huge' + '\n' + r'\centering' + '\n' + r'\caption{ ' + f'{caption}' + r' }' + '\n' + r'\label{' + f'{label}' + r'}' + '\n' + r'\begin{adjustbox}{width=1.0\linewidth, keepaspectratio}'


        if table_type == 'average_loss':
            # open tabular
            table_string+= r'\begin{tabular}{' + 'lll' * n_columns + r'} \toprule' + '\n'
            for i in range(n_columns-1):
                table_string+= r'\bf Method & \multicolumn{2}{c}{\bf Average loss (rank)} & '
            table_string+= r'\bf Method & \multicolumn{2}{c}{\bf Average loss (rank)} \\ \midrule' + '\n'
            # Wat ik wil in deze tabel:
            # Dikgedrukt de laagste average loss
            # nummering. naam & score (relatieve ranking 1st)
            # kleur (optioneel, moet ik nog over nadenken)
            bold_method = self.average_loss.idxmin()
            ranking = self.average_loss.rank(method= 'min')

            for index, method_name in enumerate(self.average_loss.keys(), 1):
                # making the best method bold
                if method_name == bold_method:
                    table_string += f'{index}. \\bf {change_name(method_name)}'
                else:
                    table_string += f'{index}. {change_name(method_name)}'

                table_string += ' & '

                if method_name == bold_method:
                    table_string += f'\\bf {self.average_loss[method_name]:.2e} & \\bf (' + r'\nth{' + f'{int(ranking[index - 1])}' + r'})'
                else:
                    table_string += f'{self.average_loss[method_name]:.2e} &  (' + r'\nth{' + f'{int(ranking[index - 1])}' + r'})'

                if index % n_columns == 0:
                    table_string += r' \\' +  '\n'
                else:
                    table_string += ' & '


        if table_type == 'top_k_sorted_average_loss':
            n_columns = 2
            k = 11
            # open tabular
            table_string+= r'\begin{tabular}{' + 'lll' * n_columns + r'} \toprule' + '\n'
            for i in range(n_columns-1):
                table_string+= r'\bf Method & \multicolumn{2}{c}{\bf Average loss (rank)} & '
            table_string+= r'\bf Method & \multicolumn{2}{c}{\bf Average loss (rank)} \\ \midrule' + '\n'
            # Wat ik wil in deze tabel:
            # Dikgedrukt de laagste average loss
            # gesorteerd naam & score (relatieve ranking 1st)
            # kleur (optioneel, moet ik nog over nadenken)
            for index, method_name in enumerate(self.sorted_average_loss.keys(), 1):
                # making the best method bold
                if index == 1:
                    table_string += f'{self.average_loss.index.get_loc(method_name)}. \\bf {change_name(method_name)}'
                else:
                    table_string += f'{self.average_loss.index.get_loc(method_name)}. {change_name(method_name)}'

                table_string += ' & '

                if index == 1:
                    table_string += f'\\bf {self.average_loss[method_name]:.2e} & \\bf (' + r'\nth{' + f'{index}' + r'})'
                else:
                    table_string += f'{self.average_loss[method_name]:.2e} & (' + r'\nth{' + f'{index}' + r'})'

                if index % n_columns == 0:
                    table_string += r' \\' +  '\n'
                else:
                    table_string += ' & '

                if index > k:
                    break


        # close the tabular
        if index % n_columns != 0:
            table_string += r' \\' +  '\n'
        table_string += r'\bottomrule' + '\n' + r'\end{tabular}' + '\n' + r'}'


        # close the table
        table_string += ' \n' + r'\end{adjustbox}' + '\n' + r'\end{table*}'

        with open(r'E:/OneDrive/PhD/Feature_Importance/Article/' + f'table_{table_type}.tex', 'w') as f:
            f.write(table_string)

        # print(table_string)

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

# results = combined_dataset_results(['results/experiment_7_individual/binary_system_2000/grouped_results/grouped-2022_07_05_14_43_09.pickle'], ['binary_system'], ALL_CLASSIFICATION)

# %%
results.sorted_average_loss
# %%
    # def _plot_loss(self):
    #     plt.plot(self.loss.keys(), self.loss.values())

# # %%
# # a = dataset_results('results/hiring_system_200-2022_05_27_21_18_22.pickle', '_')
# # result_obtained = dataset_results('results/decimal_system_2000.pickle', 'decimal_system')
# # result_obtained = dataset_results('results/binary_system_200-2022_06_01_00_54_30.pickle', 'binary_system')
# # result_obtained = dataset_results('results/experiment_6/prob_selected_025_2000-2022_06_27_14_48_12.pickle', 'prob_selected_025')
# result_obtained = dataset_results('results/experiment_7_individual/cloned_decimal_system_2000/grouped_results/grouped-2022_06_28_22_01_15.pickle', 'decimal_system')
# # result_obtained = dataset_results('results/binary_system_200/results_after_533.pickle', 'binary_system')

# # a,b = load_results('results/decimal_system_200.pickle', ['result_dict', 'time_dict'])
# # %%
# # for key in result_obtained.test_methods:
# #     print(key)

# # %%
# pd_result_loss = pd.DataFrame(result_obtained.loss, index= [0]).T
# # pd_result_loss.rename(columns= {0:'loss'})
# pd_result_loss.set_axis(['loss'], axis= 1, inplace= True)
# pd_result_loss.sort_values(by = 'loss', na_position= 'last', inplace= True)
# # %%
# pd_result_loss.head(100)
# # %%
# # # %%
# # sorted(a.grouped_loss.items())
# # # %%
# # np.sort([i for i in a.grouped_loss.keys()])
# # # %%
# # sorted(a.grouped_loss)
# # %%
# # import glob
# # for path in glob.glob('results/experiment_4/*.pickle'):
# #     print(path)
# #     result_obtained = dataset_results(path, 'None')
# #     if 'bp_feature_importance' in result_obtained.not_finished_in_time:
# #         print("ja")
# # %%
# with open('results/experiment_7_individual/cloned_decimal_system_2000/grouped_results/grouped-2022_06_28_22_01_15.pickle', 'rb') as f:
#     a = pickle.load(f)
# a[2]
# a[7]
# %%


# %%


def create_table_fi_methods(xstring, start_counter = 1):
    split_string = xstring.split(',')
    new_string = r'\tikzset{shift = {(0,\ysh)}}' + '\n' + r'\node[rectangle, draw, align = center] at (0,0){' + '\n' + r'{\bf sklearn} \\[0.5em]  \footnotesize' + '\n' + r'\begin{tabular}{llll}' + '\n'

    for j,t in enumerate(split_string):
        t = t.replace('\item ', '')
        i = j + start_counter
        new_string += f'{i}. {t} '
        if j % 4 != 3:
            new_string += '& '
        else:
            new_string += r'\\' + '\n'

    new_string += r'\end{tabular}' + '\n' + r'}'

    print(new_string)

# %%
# create_table_fi_methods(r'\item AdaBoost Classifier,\itemcolorthree{Random Forest Classifier},\itemcolorthree{Extra Trees Classifier},\item Gradient Boosting Classifier,\item SVR absolute weights,\item EL absolute weights,\itemcolorone{Permutation Importance Classifier},\item PCA sum,\item PCA weighted,\item chi2,\item f classif,\item mutual info classif,\item KL divergence,\item \textsf{R} Mutual Information,\item Fisher Score,\item FeatureVec,\item \textsf{R} Varimp Classifier,\item \textsf{R} PIMP Classifier,\itemcolortwo{Treeinterpreter Classifier},\item DIFFI,\itemcolortwo{Tree Classifier},\itemcolorfour{Linear Classifier},\itemcolorone{Permutation Classifier},\itemcolorone{Partition Classifier},\itemcolorone{Sampling Classifier},\itemcolorone{Kernel Classifier},\itemcolorone{Exact Classifier},\itemcolorone{RFI Classifier},\itemcolorone{CFI Classifier},\itemcolorfive{Sum Classifier},\itemcolorfive{Weighted X Classifier},\itemcolorfive{Weighted Y Classifier},\item f oneway,\item alexandergovern,\item pearsonr,\item spearmanr,\item pointbiserialr,\item kendalltau,\item weightedtau,\item somersd,\item linregress,\item siegelslopes,\item theilslopes,\item multiscale graphcorr,\item weight,\item gain,\item cover,\item snn,\item knn,\item bayesglm,\item lssvmRadial,\item rocc,\item ownn,\item ORFpls,\item rFerns,\item treebag,\item RRF,\item svmRadial,\item ctree2,\item evtree,\item pda,\item rpart,\item cforest,\item svmLinear,\item xyf,\item C5.0Tree,\item avNNet,\item kknn,\item svmRadialCost,\item gaussprRadial,\item FH.GBML,\item svmLinear2,\item bstSm,\item LogitBoost,\item wsrf,\item plr,\item xgbLinear,\item rf,\item null,\item protoclass,\item monmlp,\item Rborist,\item mlpWeightDecay,\item svmRadialWeights,\item mlpML,\item ctree,\item loclda,\item sdwd,\item mlpWeightDecayML,\item svmRadialSigma,\item bstTree,\item dnn,\item ordinalRF,\item pda2,\item BstLm,\item RRFglobal,\item mlp,\item rpart1SE,\item pcaNNet,\item ORFsvm,\item parRF,\item rpart2,\item gaussprPoly,\item C5.0Rules,\item rda,\item rbfDDA,\item multinom,\item gaussprLinear,\item svmPoly,\item knn,\item treebag,\item RRF,\item ctree2,\item evtree,\item pda,\item rpart,\item cforest,\item xyf,\item C5.0Tree,\item kknn,\item gaussprRadial,\item LogitBoost,\item wsrf,\item xgbLinear,\item rf,\item null,\item monmlp,\item Rborist,\item mlpWeightDecay,\item mlpML,\item ctree,\item mlpWeightDecayML,\item dnn,\item pda2,\item RRFglobal,\item mlp,\item rpart1SE,\item parRF,\item rpart2,\item gaussprPoly,\item C5.0Rules,\item rbfDDA,\item multinom,\item gaussprLinear,\item binaryConsistency,\item chiSquared,\item cramer,\item gainRatio,\item giniIndex,\item IEConsistency,\item IEPConsistency,\item mutualInformation,\item roughsetConsistency,\item ReliefFeatureSetMeasure,\item symmetricalUncertain,\itemcolorone{IteratedEstimator},\itemcolorone{PermutationEstimator},\itemcolorone{KernelEstimator},\itemcolorone{SignEstimator},\itemcolorone{Shapley},\itemcolorone{Banzhaf},\item RF,\itemcolorsix{Garson},\itemcolorsix{VIANN},\itemcolorsix{LOFO},\item Relief,\item ReliefF,\item RReliefF,\item fit criterion measure,\item f ratio measure,\item gini index,\item su measure,\item spearman corr,\item pearson corr,\item fechner corr,\item kendall corr,\item chi2 measure,\item anova,\item laplacian score,\item information gain,\item modified t score MIM,\item MRMR,\item JMI,\item CIFE,\item CMIM,\item ICAP,\item DCSF,\item CFR,\item MRI,\item IWFS,\item NDFS,\item RFS,\item SPEC,\item MCFS,\item UDFS')
# %%


# X, Y, labelencoded_Y, onehotencoded_Y, dataset, result_dict, time_dict, did_not_work, not_finished_in_time, time_limit, test_methods = group_together_results(target_path= 'results/experiment_7_individual/binary_system_2000/grouped_results/grouped', results_after_path='results/experiment_7/backup/binary_system_2000/results_after_52.pickle', path_names=glob.glob('results/experiment_7_individual/binary_system_2000/*.pickle'), save = True)
# # %%
# for i, name in enumerate(list_of_all_methods):
#     if name not in result_dict and name not in did_not_work and name not in not_finished_in_time:
#         print(i)
# # %%
# list_of_all_methods[23]

# %%
# data_name = 'max_function_expanding_dist_5_2000'
# group_together_results(target_path= f'results/experiment_7_individual/{data_name}/grouped_results/grouped', path_names=glob.glob(f'results/experiment_7_individual/{data_name}/*.pickle'), save = True)

# grouped_results = glob.glob(f'results/experiment_7_individual/{data_name}/grouped_results/*.pickle')
# latest_file = max(grouped_results, key=os.path.getctime)

# result_obtained = dataset_results(latest_file, '', FAST_LIST)

# # %%
# pd_result_loss = pd.DataFrame(result_obtained.loss, index= [0]).T
# # pd_result_loss.rename(columns= {0:'loss'})
# pd_result_loss.set_axis(['loss'], axis= 1, inplace= True)
# pd_result_loss.sort_values(by = 'loss', na_position= 'last', inplace= True)
# pd_result_loss.head(100)
# %%
# pd_result_time = pd.DataFrame(result_obtained.time_dict, index= [0]).T
# # pd_result_time.rename(columns= {0:'time'})
# pd_result_time.set_axis(['time'], axis= 1, inplace= True)
# pd_result_time.sort_values(by = 'time', na_position= 'last', inplace= True)
# pd_result_time.head(300)
# # %%
# for i in result_obtained.did_not_work:
#     print(f"'{i}',")
# for i,j in pd_result_time.iterrows():
#     print(f"'{i}',")

# %%

# %%
# j = -1
# l = 0
# for i in dataset_names:
#     j +=1
#     k = j % 4
#     print(f'{k}. {i} {l}')
#     if k == 3:
#         l += 1
# # %%
# result_dataset_paths = next(os.walk('results/experiment_7_individual/'))[1]
# result_dataset_paths.remove('backup_2')
# %%

# %%
