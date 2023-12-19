import numpy as np
from perceptron import Perceptron
import matplotlib.pyplot as plt

N_FEATURES = 10
N_SAMPLES = 40

true_weights = np.random.normal(scale=2, size=(N_FEATURES, ))
true_bias = np.random.normal()
X = np.random.normal(scale=2, size=(N_SAMPLES, N_FEATURES))
Y = np.sign(X@true_weights + true_bias)
perc = Perceptron(N_FEATURES)
accs = perc.fit(X, Y)

plt.plot([i+1 for i in range(len(accs))], accs)
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Percentage accuracy", loc="center")
plt.show()