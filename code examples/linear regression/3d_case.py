import numpy as np
import matplotlib.pyplot as plt

N_POINTS = 20
w_0 = 2
w1, w2 = 1.2, 1
w_vec = np.array([w1, w2])

X = np.random.normal(scale=1, loc=1, size=(N_POINTS, 2))
X = X - np.min(X)

# create a linear relationship and add some noise/error
y = X@w_vec + w_0 + np.random.normal(scale=1, size=(N_POINTS))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, c='red', s=20)

Xs = np.arange(np.min(X[:, 0]), np.max(X[:, 0]), 0.25)
Ys = np.arange(np.min(X[:, 1]), np.max(X[:, 1]), 0.25)
Xs, Ys = np.meshgrid(Xs, Ys)
Z = Xs*w1 + Ys*w2 + w_0
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$y$')
ax.plot_surface(Xs, Ys, Z, vmin=Z.min() * 2, alpha=0.8, cmap='viridis')
plt.show()