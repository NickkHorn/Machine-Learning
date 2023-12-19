import numpy as np
from logistic_regression import BinaryLR
import matplotlib.pyplot as plt

N_FEATURES = 3
N_SAMPLES = 40

true_weights = np.random.normal(scale=2, size=(N_FEATURES, ))
true_bias = np.random.normal()
X = np.random.normal(scale=2, size=(N_SAMPLES, N_FEATURES))
Y = np.where(X@true_weights + true_bias >= 0, 1, 0)
model = BinaryLR(N_FEATURES, lr=5e-4)
losses, accs = model.fit(X, Y, n_iters=100)

plt.plot([epoch for epoch in range(len(losses))], losses)
plt.xlabel("Epoca")
plt.ylabel("Funzione di costo")
plt.show()

plt.plot([epoch for epoch in range(len(accs))], accs)
plt.xlabel("Epoca")
plt.ylabel("Accuratezza")
plt.show()


positive_indices = np.where(Y == 1)
negative_indices = np.where(Y == 0)

if N_FEATURES == 3:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X[positive_indices, 0], X[positive_indices, 1], X[positive_indices, 2], c='red', marker='+', s=20, label='Classe positiva')
    ax.scatter(X[negative_indices, 0], X[negative_indices, 1], X[negative_indices, 2], c='c', marker='$-$', s=20, label='Classe negativa')

    Xs = np.arange(np.min(X[:, 0]), np.max(X[:, 0]), 0.25)
    Ys = np.arange(np.min(X[:, 1]), np.max(X[:, 1]), 0.25)
    Xs, Ys = np.meshgrid(Xs, Ys)
    # -(w1*x + w2*y + w0)/w3 = z
    Z = -(model.weights[0]*Xs + model.weights[1]*Ys + model.w0)/model.weights[2]
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')
    ax.legend()
    ax.plot_surface(Xs, Ys, Z, vmin=Z.min() * 2)
    plt.show()

elif N_FEATURES == 2:
    plt.scatter(X[positive_indices, 0], X[positive_indices, 1], c='red', marker='+', label='Classe positiva')
    plt.scatter(X[negative_indices, 0], X[negative_indices, 1], c='c', marker='$-$', label='Classe negativa')

    Xs = np.arange(np.min(X[:, 0]), np.max(X[:, 0]), 0.25)
    Ys = (-model.weights[0]*Xs + model.w0)/model.weights[1]
    plt.plot(Xs, Ys, color="black", label='Confine di decisione')
    plt.xlabel("$x_1$")
    plt.xlabel("$x_2$")
    plt.legend()
    plt.show()