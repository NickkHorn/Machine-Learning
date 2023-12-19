import numpy as np

class Adaline:

    def __init__(self, n_inputs, lr=0.001):
        self.n_inputs = n_inputs
        self.lr = lr

        self.weights = np.random.normal(size=(n_inputs, ))
        self.w0 = np.random.normal() # bias

    def predict(self, X):
        return np.sign(X@self.weights + self.w0)
    
    def activate(self, X):
        return X@self.weights + self.w0

    def loss(self, y, y_pred):
        return np.sum((y-y_pred)**2)/2

    def fit(self, X, y, n_epochs=100):
        losses = []
        accs = [] # accuracy (%)
        n_samples = len(y)
        for n in range(n_epochs):
            y_pred = self.activate(X)

            # LOSS AND STATISTICS CALCULATION
            output = np.sign(y_pred)
            loss = self.loss(y, y_pred)
            acc = 100*(n_samples - np.count_nonzero(output-y))/n_samples
            accs.append(acc)
            print(f"[Epoch {n+1}] Loss: {loss}   |   Accuracy: {acc}")
            losses.append(loss)

            # WEIGHTS UPDATE
            E = y - y_pred
            dw = self.lr * X.T@E
            self.weights = self.weights + dw
            self.w0 += self.lr * np.sum(E)

        return losses, accs