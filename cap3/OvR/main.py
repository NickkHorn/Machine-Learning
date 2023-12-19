import numpy as np
import matplotlib.pyplot as plt
from ovr import Ovr
from perceptron import Perceptron

N_FEATURES = 3
N_SAMPLES = 40

X = np.random.normal(scale=1, size=(N_SAMPLES, N_FEATURES))
X_rows = [row for row in X]
y = []
for row in X_rows:
    val = np.sum(row)
    if -0.3 <= val <= 0.3:
        y.append(1)
    else:
        y.append(2 if val > 0.3 else 0)

y = np.array(y)
ovr = Ovr()
classifiers = [Perceptron(N_FEATURES), Perceptron(N_FEATURES), Perceptron(N_FEATURES, lr=1e-3)]
clf_accuracies = ovr.fit(X, y, classifiers=classifiers, n_epochs=50, verbose=False)

colors = ['red', 'green', 'blue']
max_epochs = 0
for i, clf_accuracy in enumerate(clf_accuracies):
    clf_n_epochs = len(clf_accuracy)
    plt.plot([n for n in range(1, len(clf_accuracy)+1)], clf_accuracy, color=colors[i], label=f"classificatore classe #{i}")
    if clf_n_epochs > max_epochs:
        max_epochs = clf_n_epochs

acc = ovr.get_accuracy(y, ovr.predict(X))
plt.plot([n for n in range(1, max_epochs+1)], np.ones(shape=(max_epochs,))*acc, color='c', label=f"classificatore ovr #{i}")
plt.xlabel("Epoca")
plt.ylabel("Accuratezza")
plt.legend()
plt.show()