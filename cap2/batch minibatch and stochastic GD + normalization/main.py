import numpy as np
from adaline import *
import matplotlib.pyplot as plt

def plot_model_loss(losses, ax, title):
    ax.plot([i+1 for i in range(len(losses))], losses)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

def plot_model_accs(accs, ax, title):
    ax.plot([i+1 for i in range(len(accs))], accs)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")

def normalize(X):
    # per ogni colonna
    for i in range(X.shape[1]):
        col = X[:, i]
        X[:, i] = (col - np.mean(col))/np.var(col)
    return X

N_FEATURES = 10
N_SAMPLES = 40

true_weights = np.random.normal(scale=2, size=(N_FEATURES, ))
true_bias = np.random.normal()
X = np.random.normal(scale=2, size=(N_SAMPLES, N_FEATURES))
X_norm = normalize(X)
Y = np.sign(X@true_weights + true_bias)

batch = AdalineBatch(N_FEATURES)
batch_losses, batch_accs = batch.fit(X, Y)
batch = AdalineBatch(N_FEATURES)
batch_normalized_losses, batch_normalized_accs = batch.fit(X_norm, Y)

minibatch = AdalineMB(N_FEATURES)
minibatch_losses, minibatch_accs = minibatch.fit(X, Y, batch_size=16)
minibatch = AdalineMB(N_FEATURES)
minibatch_normalized_losses, minibatch_normalized_accs = minibatch.fit(X_norm, Y, batch_size=16)

stochastic = AdalineStochastic(N_FEATURES)
stochastic_losses, stochastic_accs = stochastic.fit(X, Y)
stochastic = AdalineStochastic(N_FEATURES)
stochastic_normalized_losses, stochastic_normalized_accs = stochastic.fit(X_norm, Y)

fig, ax = plt.subplots(3, 2)
plot_model_loss(batch_losses, ax[0,0], "GD a Lotti")
plot_model_loss(minibatch_losses, ax[1,0], "Mini-Batch GD")
plot_model_loss(stochastic_losses, ax[2,0], "GD Stocastica")

plot_model_loss(batch_normalized_losses, ax[0,1], "GD a Lotti + Normalization")
plot_model_loss(minibatch_normalized_losses, ax[1,1], "Mini-Batch GD + Normalization")
plot_model_loss(stochastic_normalized_losses, ax[2,1], "GD Stocastica + Normalization")
fig.suptitle("Losses")
plt.show()


fig, ax = plt.subplots(3, 2)
plot_model_loss(batch_accs, ax[0,0], "GD a Lotti")
plot_model_loss(minibatch_accs, ax[1,0], "Mini-Batch GD")
plot_model_loss(stochastic_accs, ax[2,0], "GD Stocastica")

plot_model_loss(batch_normalized_accs, ax[0,1], "GD a Lotti + Normalization")
plot_model_loss(minibatch_normalized_accs, ax[1,1], "Mini-Batch GD + Normalization")
plot_model_loss(stochastic_normalized_accs, ax[2,1], "GD Stocastica + Normalization")
fig.suptitle("Accuracies")
plt.show()
