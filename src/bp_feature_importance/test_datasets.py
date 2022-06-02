# %%
import numpy as np
import random
# %%
n_samples = 20000
X = np.random.uniform(low= 0, high= 1, size= 3 * n_samples)
X_1 = X[:n_samples]
X_2 = X[n_samples:2*n_samples]
X_3 = X[2*n_samples:]
Y = np.maximum(X_1, X_2, X_3)
amax = np.argmax(np.stack((X_1,X_2,X_3), axis =1), axis = 1)
# %%
print(X_1)
print(X_2)
print(X_3)