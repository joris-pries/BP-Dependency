# Copied from https://github.com/gcskoenig/icpr2020-rfi/blob/master/rfi.py
# Is used to determine relative feature importance

import numpy as np
from DeepKnockoffs import KnockoffMachine
from DeepKnockoffs import GaussianKnockoffs
import copy
from scipy.stats import t

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
rc('text', usetex=True)
sns.set_style("whitegrid", {'ytick.left': False, 'ytick.right': False, 
	'axes.grid': True, 'axes.spines.left': False, 'axes.spines.top':False, 
	'axes.spines.right': False, 'axes.spines.bottom': False, 
	'axes.axisbelow': True}) 
sns.set_context('paper')




def paired_t(rfis, lss):
	'''
	based on Conditional Predictive Impact, Watson and Wright 2019

	Args:
		rfis: n_repeat, d
		lss: n_repeat, n, d
	'''
	n = lss.shape[1]
	d = rfis.shape[1]
	n_repeat = rfis.shape[0]

	rfis = rfis.reshape((n_repeat, 1, d))
	s = (lss - rfis) # should be broadcasted correctly
	s = np.sqrt((1/(n-1))*np.sum(np.power(s, 2), axis=1))
	se = s / np.sqrt(n)
	rfis = rfis.reshape((n_repeat, d))
	ts = rfis / se
	pvals = 1 - t.cdf(ts, (n - 1))
	return pvals


def deep_knockoff_caller(X_train, X_test):
	# TODO
	return None


def knockoff_caller(X_train, X_test):
	'''
	knockoff code as given in examples for deepknockoff paper

	'''
	SigmaHat = np.cov(X_train, rowvar=False)
	second_order = GaussianKnockoffs(SigmaHat, mu=np.mean(X_train,0))#, method="sdp")
	knockoffs = second_order.generate(X_test)
	return knockoffs


def create_2nd_order_knockoff(j, G, X_train, X_test):
	'''
	j: index of variable of interest
	G: list of conditioning indexes
	'''
	G = np.array(G) # conditioning set
	S = np.zeros(np.prod(G.shape)+1, dtype=np.int16)
	S[:-1] = G
	S[-1] = j # variable to be replaced
	knockoffs = knockoff_caller(X_train[:, S], X_test[:, S]) # creates knockoffs
	knockoff_j = knockoffs[:, -1]
	return knockoff_j # knockoff of j computed from G


def create_permuted(X_test, D):
	'''
	returns permuted versions of all variables in D
	'''
	P_test = np.zeros((X_test.shape[0], D.shape[0]))
	for kk in np.arange(0, D.shape[0], 1):
		p_xj = copy.deepcopy(X_test[:, kk])
		p_xj = np.random.permutation(p_xj)
		P_test[:, kk] = p_xj
	return P_test


def create_knockoffs(G, X_train, X_test, D):
	'''
	leverages model-x-knockoffs to create replacement variables

	args:
		G: variables to condition on
		X_train: training dataset
		X_test: prediction dataset
		D: the variables for which knockoffs shall be generated

	returns
		K_test: (#n_test, |D|)-matrix with knockoffs

	'''
	K_test = np.zeros((X_test.shape[0], D.shape[0]))
	for kk in np.arange(0, D.shape[0], 1):
		k_xj = np.zeros((X_train.shape[0], 1))
		if D[kk] in G:
			k_xj = copy.deepcopy(X_test[:,D[kk]]) # if the variable is in G, simply return a copy
		else:
			k_xj = create_2nd_order_knockoff(D[kk], G, X_train, X_test) # create 2nd-order knockoff
		K_test[:, kk] = k_xj
	return K_test


def rfi(predict, loss, G, X_train, X_test, y_test, D, n_repeats=20, return_perturbed=False):
	'''
	Relative feature importance function.

	Args:
		predict: prediction function with arguments X_D
		loss: elementswise loss function that returns a float
		G: relative feature importance conditioning set
		X_train: training dataset
		X_test: test dataset
		y_test: test predictions
		D: variable indexes for model training (can be used to prepare input for predict)
		n_repeats: number of iterations for estimate
		return_perturbed: return the perturbed datasets
	'''
	# initialize relative feature importance array
	n = y_test.shape[0]
	d = D.shape[0]

	# compute large-sample test
	lss = np.zeros((n_repeats, n, d)) # n_repeats: iterations; #n: number of observations in test, D.shape: # variables in model
	perturbed = [] # array to collect perturbed versions of dataset
	for nn in np.arange(0, n_repeats, 1): # repeatedly compute rfi (n_repeats times)
		T_test = np.zeros((n, d), dtype=np.float64) #\tilde{X}_test empty array
		if G.shape[0]==0: # if conditionig set is empty we can simply permute
			T_test = create_permuted(X_test, D)
		else: # otherwise compute knockoffs
			T_test = create_knockoffs(G, X_train, X_test, D) # TODO generalize for other choices of knockoffs
		if return_perturbed:
			perturbed.append(T_test) # return perturbed versions if specified
		ls_ests = np.zeros((n, d)) # estimated losses (n number of observations, D number of variable)
		y_pred = predict(X_test[:, D]) # estimated predictions
		ls_est = loss(y_test, y_pred).reshape(-1, 1) # baseline loss, observation-wise (n, 1)

		for kk in np.arange(0, d, 1): # iterate over variables
			X_modified = np.array(X_test[:, D], copy=True) 
			X_modified[:, kk] = T_test[:, kk] # replace the variable with the respective knockoff
			y_pred_kk = predict(X_modified) # predict on the knockoff dataset
			ls_est_kk = loss(y_test, y_pred_kk) # observation-wise loss
			ls_ests[:, kk] = ls_est_kk # save losses

		ls_new = ls_ests - ls_est # compute repective differences (n, d) - (n, 1) -> (n, d)
		lss[nn, :, :] = ls_new # save in array

	rfis = np.mean(lss, axis=1) # compute rfis

	if return_perturbed:
		return np.mean(rfis, axis=0), np.std(rfis, axis=0), rfis, lss, perturbed
	else:
		return np.mean(rfis, axis=0), np.std(rfis, axis=0), rfis, lss


def cfi(predict, loss, X_train, X_test, y_test, D, n_repeats=20, return_perturbed=False, Rs=[]):
	n = y_test.shape[0]
	d = D.shape[0]

	Rs = np.array(Rs)
	R = np.zeros((D.shape[0]+Rs.shape[0]), dtype=np.int16)
	R[:d] = D
	R[d:] = Rs

	lss = np.zeros((n_repeats, n, d))
	perturbed = []
	for nn in np.arange(0, n_repeats, 1):
		T_test = knockoff_caller(X_train[:, R], X_test[:, R])[:, :d] #also use Rs for knockoffs
		if return_perturbed:
			T_test.append(perturbed)

		ls_ests = np.zeros((n, d))
		y_pred = predict(X_test[:, D])
		ls_est = loss(y_test, y_pred).reshape(-1, 1)

		for kk in np.arange(0, d, 1):
			X_modified = np.array(X_test[:, D], copy=True)
			X_modified[:, kk] = T_test[:, kk]
			y_pred_kk = predict(X_modified)
			ls_est_kk = loss(y_test, y_pred_kk)
			ls_ests[:, kk] = ls_est_kk

		ls_new = ls_ests - ls_est # compute repective differences (n, d) - (n, 1) -> (n, d)
		lss[nn, :, :] = ls_new # save in array

	cfis = np.mean(lss, axis=1)

	if return_perturbed:
		return np.mean(cfis, axis=0), np.std(cfis, axis=0), cfis, lss, perturbed
	else:
		return np.mean(cfis, axis=0), np.std(cfis, axis=0), cfis, lss


def plot_rfis(rfis, fnames, rfinames, savepath, figsize=(16,10), textformat='{:5.2f}'):
    '''
    rfis: list of tuples (means, stds)

    '''

    ind = np.arange(len(fnames))  # the x locations for the groups
    width = (1/len(rfinames)*0.95)  # the width of the bars

    fig, ax = plt.subplots()
    rects = []
    for rfi_ind in np.arange(0, len(rfinames), 1):
        print(rfi_ind)
        rects_inx = ax.bar(ind + width*(rfi_ind+0.5), rfis[rfi_ind][0], width, 
        	yerr=rfis[rfi_ind][1], label=rfinames[rfi_ind])
        rects.append(rects_inx)

    ax.set_ylabel('Importance')
    ax.set_title('RFIs Plot')
    ax.set_xticks(ind)
    ax.set_xticklabels(fnames)
    ax.legend()


    def autolabel(rects, xpos=0):
        """
        Attach a text label above each bar in *rects*, displaying its height.
        """
        for rect in rects:
            height = rect.get_height()
            ax.annotate(textformat.format(height),
                        xy=(rect.get_x(), height),
                        xytext=(3, 4),  # use 3 points offset 
                        #previously in xpos of xytext: +rect.get_width()/2
                        textcoords="offset points",  # in both directions
                        va='bottom')


    for ii in range(len(rects)):
        autolabel(rects[ii], xpos=ii)

    fig.tight_layout()
    plt.savefig(savepath, figsize=figsize)
    plt.show()