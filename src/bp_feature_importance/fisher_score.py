# Copied form https://www.codestudyblog.com/cs2112pyc/1223230432.html
# Used to determine the fisher score

''' calculate fisher score'''

import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class fisherscore():
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.nSample, self.nDim = X.shape
        self.labels = np.unique(y)
        self.nClass = len(self.labels)
        self.total_mean = np.mean(self.X, axis=0)
        '''
        [mean(a1), mean(a2), ... , mean(am)]
        '''
        self.class_num, self.class_mean, self.class_std = self.get_mean_std()
        '''
        std(c1_a1), std(c1_a2), ..., std(c1_am)
        std(c2_a1), std(c2_a2), ..., std(c2_am)
        std(c3_a1), std(c3_a2), ..., std(c3_am)
        '''
        self.fisher_score_list = [self.cal_FS(j) for j in range(self.nDim)]


    def get_mean_std(self):
        Num = np.zeros(self.nClass)
        Mean = np.zeros((self.nClass, self.nDim))
        Std = np.zeros((self.nClass, self.nDim))
        for i, lab in enumerate(self.labels):
            idx_list = np.where(self.y == lab)[0]
            # print(idx_list[0])
            Num[i] = len(idx_list)
            Mean[i] = np.mean(self.X[idx_list], axis=0)
            Std[i] = np.std(self.X[idx_list], axis=0)
        return Num, Mean, Std

    def cal_FS(self,j):
        Sb_j = 0.0
        Sw_j = 0.0
        for i in range(self.nClass):
            Sb_j += self.class_num[i] * (self.class_mean[i,j] - self.total_mean[j])**2
            Sw_j += self.class_num[i] * self.class_std[i,j] **2
        return Sb_j / Sw_j