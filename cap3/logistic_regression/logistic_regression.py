import numpy as np

# Regressione logistica a batch per classificazione binaria
class BinaryLR:

    def __init__(self, n_features, lr=0.001):
        self.n_features = n_features
        self.lr = lr
        self.weights = np.random.normal(scale=0.01, size=(n_features, ))
        self.w0 = np.random.normal(scale=0.01)

    def predict(self, X):
        # analogo a 1 se self.sigmoid(X@self.weights + self.w0) >= 0.5 else -1
        return np.where(X@self.weights + self.w0 >= 0, 1, 0)

    def sigmoid(self, x):
        return 1/(1+np.exp(np.clip(-x, -250, 250)))

    def activate(self, X):
        return self.sigmoid(X@self.weights + self.w0)

    def __accuracy(self, y, y_activation):
        n_samples = len(y)
        classifications = np.where(y_activation >= 0.5, 1, 0)
        num_wrong = np.count_nonzero(classifications - y)
        return (n_samples - num_wrong)/n_samples

    def loss(self, y, y_activation):
        return -np.sum(y*np.log(y_activation) + (1-y)*np.log(1-y_activation))

    def fit(self, X, y, n_iters=10):
        losses, accs = [], []
        for n in range(n_iters):
            y_activation = self.activate(X)
            losses.append(self.loss(y, y_activation))
            accs.append(self.__accuracy(y, y_activation))

            errors = y-y_activation
            self.weights = self.weights + self.lr * X.T.dot(errors)
            self.w0 += self.lr*np.sum(errors)
        
        return losses, accs
