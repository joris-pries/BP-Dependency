
# %%
# Modified version of https://github.com/rebelosa/feature-importance-neural-networks/blob/master/variance-based%20feature%20importance%20in%20artificial%20neural%20networks.ipynb to extract all fi measures

import tensorflow
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils, to_categorical
from keras import optimizers
# from keras.layers.advanced_activations import PReLU
# from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from sklearn import datasets
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, scale
from keras.utils import np_utils
import keras

# %%
def NN1(input_dim, output_dim, isClassification = True):
    print("Starting NN1")

    model = Sequential()
    model.add(Dense(50, input_dim=input_dim, activation='linear', kernel_initializer='normal', kernel_regularizer=l2(0.01)))
    model.add(Dense(100, activation='linear', kernel_initializer='normal', kernel_regularizer=l2(0.01)))
    model.add(Dense(50, activation='linear', kernel_initializer='normal', kernel_regularizer=l2(0.01)))

    if (isClassification == False):
        model.add(Dense(1, kernel_initializer='normal'))
        model.compile(loss='mean_squared_error', optimizer='sgd')
    elif (isClassification == True):
        model.add(Dense(output_dim, activation='softmax', kernel_initializer='normal'))
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    return model

# %%
def NN2(input_dim, output_dim, isClassification = True):
    print("Starting NN2")

    model = Sequential()
    model.add(Dense(50, input_dim=input_dim, activation='relu', kernel_initializer='normal', kernel_regularizer=l2(0.01)))
    model.add(Dense(100, activation='relu', kernel_initializer='normal', kernel_regularizer=l2(0.01)))
    model.add(Dense(50, activation='relu', kernel_initializer='normal', kernel_regularizer=l2(0.01)))

    if (isClassification == False):
        model.add(Dense(1, kernel_initializer='normal'))
        model.compile(loss='mean_squared_error', optimizer='sgd')
    elif (isClassification == True):
        model.add(Dense(output_dim, activation='softmax', kernel_initializer='normal'))
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    return model
# %%
# VIANN
# Variance-based Feature Importance of Artificial Neural Networks
class VarImpVIANN(keras.callbacks.Callback):
    def __init__(self, verbose=0):
        self.verbose = verbose
        self.n = 0
        self.M2 = 0.0

    def on_train_begin(self, logs={}, verbose = 1):
        if self.verbose:
            print("VIANN version 1.0 (Wellford + Mean) update per epoch")
        self.diff = self.model.layers[0].get_weights()[0]

    def on_epoch_end(self, batch, logs={}):
        currentWeights = self.model.layers[0].get_weights()[0]

        self.n += 1
        delta = np.subtract(currentWeights, self.diff)
        self.diff += delta/self.n
        delta2 = np.subtract(currentWeights, self.diff)
        self.M2 += delta*delta2

        self.lastweights = self.model.layers[0].get_weights()[0]

    def on_train_end(self, batch, logs={}):
        if self.n < 2:
            self.s2 = float('nan')
        else:
            self.s2 = self.M2 / (self.n - 1)

        scores = np.sum(np.multiply(self.s2, np.abs(self.lastweights)), axis = 1)

        self.varScores = (scores - min(scores)) / (max(scores) - min(scores))
        if self.verbose:
            print("Most important variables: ",
                  np.array(self.varScores).argsort()[-10:][::-1])
# %%
# Taken from https://csiu.github.io/blog/update/2017/03/29/day33.html
def garson(A, B):
    """
    Computes Garson's algorithm
    A = matrix of weights of input-hidden layer (rows=input & cols=hidden)
    B = vector of weights of hidden-output layer
    """
    B = np.diag(B)

    # connection weight through the different hidden node
    cw = np.dot(A, B)

    # weight through node (axis=0 is column; sum per input feature)
    cw_h = abs(cw).sum(axis=0)

    # relative contribution of input neuron to outgoing signal of each hidden neuron
    # sum to find relative contribution of input neuron
    rc = np.divide(abs(cw), abs(cw_h))
    rc = rc.sum(axis=1)

    # normalize to 100% for relative importance
    ri = rc / rc.sum()
    return(ri)
# %%
# Adapted from https://csiu.github.io/blog/update/2017/03/29/day33.html
class VarImpGarson(keras.callbacks.Callback):
    def __init__(self, verbose=0):
        self.verbose = verbose

    def on_train_end(self, batch, logs={}):
        if self.verbose:
            print("VarImp Garson")
        """
        Computes Garson's algorithm
        A = matrix of weights of input-hidden layer (rows=input & cols=hidden)
        B = vector of weights of hidden-output layer
        """
        A = self.model.layers[0].get_weights()[0]
        B = self.model.layers[len(self.model.layers)-1].get_weights()[0]

        self.varScores = 0
        for i in range(B.shape[1]):
            self.varScores += garson(A, np.transpose(B)[i])
        if self.verbose:
            print("Most important variables: ",
                np.array(self.varScores).argsort()[-10:][::-1])
# Leave-One-Feature-Out LOFO
def LeaveOneFeatureOut(model, X, Y):
    OneOutScore = []
    n = X.shape[0]
    for i in range(0,X.shape[1]):
        newX = X.copy()
        newX[:,i] = 0 #np.random.normal(0,1,n)
        OneOutScore.append(model.evaluate(newX, Y, batch_size=2048, verbose=0))
    OneOutScore = pd.DataFrame(OneOutScore[:])
    ordered = np.argsort(-OneOutScore.iloc[:,0])
    return(OneOutScore, ordered)
# %%
from keras.callbacks import Callback
import numpy as np

class AccuracyMonitor(Callback):
    def __init__(self,
                 monitor=['val_acc'],
                 verbose=0,
                 min_epochs=5,
                 baseline=None):
        super(AccuracyMonitor, self).__init__()

        self.monitor = monitor
        self.baseline = baseline
        self.verbose = verbose
        self.min_epochs = min_epochs
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs.get(self.monitor) > self.baseline and epoch > self.min_epochs:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            print('\n Stopped at epoch {epoch}. Accuracy of {accuracy} reached.'.format(epoch=(self.stopped_epoch + 1), accuracy=logs.get(self.monitor)), "\n")

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
# %%
import matplotlib.pyplot as plt
from numpy.random import seed
from keras.callbacks import EarlyStopping

def modified_runExp(method, X, Y, isClassification= False, mdl = "NN1", xseed = 42, epochs = 1000, verbose = 0):

    seed(xseed)


    if isClassification == True:
        #Classification
        labels_encoded = []
        for labels in [Y]:
            encoder = LabelEncoder()
            encoder.fit(labels)
            encoded_Y = encoder.transform(labels)
            # convert integers to dummy variables (i.e. one hot encoded)
            labels_encoded.append(np_utils.to_categorical(encoded_Y))
        YLabels = labels_encoded[0]

        if method == 'RF':
            # fit a Random Forest model to the data
            RFmodel = RandomForestClassifier(n_estimators=100)

        output_size = YLabels.shape[1]

    else:
        YLabels = scale(Y)
        output_size = 1

        if method == 'RF':
            # fit a Random Forest model to the data
            RFmodel = RandomForestRegressor(n_estimators=100)

    X = scale(X)
    Y = YLabels

    if method == 'RF':
        RFmodel.fit(X, Y)
        return RFmodel.feature_importances_

    if method == 'VIANN':
        VIANN = VarImpVIANN(verbose=verbose)
        clbs = [VIANN]
    if method == 'Garson':
        Garson = VarImpGarson(verbose=verbose)
        clbs = [Garson]

    if method =='LOFO':
        clbs = []

    if (mdl == "NN1"):
        model = NN1(X.shape[1], output_size, isClassification)
    elif (mdl == "NN2"):
        model = NN2(X.shape[1], output_size, isClassification)




    if method in ['LOFO', 'Garson', 'VIANN']:
        model.fit(X, Y, validation_split=0.05, epochs=epochs, batch_size=np.round(X.shape[0]/7).astype(int), shuffle=True,
                verbose=verbose, callbacks = clbs)

    if method == 'LOFO':
        LOFO, LOFO_Ordered = LeaveOneFeatureOut(model, X, Y)
        return LOFO

    if method == 'Garson':
        return Garson.varScores

    if method == 'VIANN':
        return VIANN.varScores



    return None


# %%
kwargs = {'n_observations': 200}
X_1 = np.random.randint(10, size=kwargs['n_observations'])
X_2 = np.random.randint(10, size=kwargs['n_observations'])
X_3 = np.random.randint(10, size=kwargs['n_observations'])
Y = X_1 + 10 * X_2 + 100 * X_3
X = np.stack((X_1, X_2, X_3), axis=1)
dataset = np.stack((X_1, X_2, X_3, Y), axis=1)

fi_results = np.array(modified_runExp(method = 'LOFO', X= X, Y= Y, isClassification= False, mdl= 'NN1'))
# %%
