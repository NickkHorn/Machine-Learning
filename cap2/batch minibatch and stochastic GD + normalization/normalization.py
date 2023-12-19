import numpy as np
import matplotlib.pyplot as plt

def normalize(X):
    # per ogni colonna
    for i in range(X.shape[1]):
        col = X[:, i]
        X[:, i] = (col - np.mean(col))/np.var(col)
    return X

N_FEATURES = 2
N_SAMPLES = 40

true_weights = np.random.normal(scale=2, size=(N_FEATURES, ))
true_bias = np.random.normal()
X = np.random.normal(scale=1.6, size=(N_SAMPLES, N_FEATURES), loc=3.0)
Y = np.random.normal(scale=1, size=(N_SAMPLES, N_FEATURES))

fig, ax = plt.subplots()
ax.scatter(X, Y, c=['green'], label='Default points')
ax.scatter(normalize(X), Y, c=['red'], label='Normalized points')
ax.set_aspect('equal')
ax.grid(True, which='both')
ax.legend()

ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')

plt.show()
