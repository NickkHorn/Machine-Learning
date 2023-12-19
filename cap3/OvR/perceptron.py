import numpy as np

class Perceptron:

    # lr è il learning rate
    def __init__(self, n_neurons, lr=0.01):
        self.n_neurons = n_neurons
        weights_stddev = 1/np.sqrt(n_neurons)
        self.weights = np.random.normal(size=(n_neurons,), scale=weights_stddev)
        self.bias = 0 # bias
        self.lr = lr

    def predict(self, X):
        return np.sign(X@self.weights + self.bias)

    def fit(self, X, Y, n_epochs=100, verbose=False):
        n_samples = len(Y)
        accs = []
        for n in range(n_epochs):
            y_pred = self.predict(X)
            n_correct = n_samples - np.count_nonzero(Y-y_pred)
            perc_acc = 100*n_correct/n_samples
            accs.append(perc_acc)
            if verbose:
                print(f"Epoch #{n} accuracy: {perc_acc}%")

            if perc_acc == 100:
                break

            # Aggiorna i pesi
            n = 0
            for xi, y in zip(X,Y):
                self.weights = self.weights + self.lr*(y-y_pred[n])*xi
                self.bias += self.lr*(y-y_pred[n])
                n = n+1
        
        return accs