import numpy as np

'''
Adaline con una discesa del gradiente a lotti
'''
class AdalineBatch:

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
    


'''
Adaline con una discesa del gradiente a mini-batch
'''
class AdalineMB:

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

    def fit(self, X, y, n_epochs=100, batch_size=32):
        losses = []
        accs = [] # accuracy (%)
        n_samples = len(y)
        for n in range(n_epochs):
            print(f"[Epoch {n+1}]")
            # per ogni mini batch
            for i in range(int(np.ceil(n_samples/batch_size))):
                mb_start_idx = i*batch_size
                mb_end_idx = min(n_samples, (i+1)*batch_size)
                mb_X = X[mb_start_idx:mb_end_idx, :]
                mb_y = y[mb_start_idx:mb_end_idx]

                # prendi il minore di X con solo le righe della mini batch corrente
                mb_y_pred = self.activate(X[mb_start_idx:mb_end_idx, :])

                # LOSS AND STATISTICS CALCULATION
                output = np.sign(mb_y_pred)
                loss = self.loss(mb_y, mb_y_pred)
                acc = 100*(n_samples - np.count_nonzero(output-mb_y))/n_samples
                print(f"\t[Mini batch #{i+1}]: Samples interval: [{mb_start_idx}, {mb_end_idx-1}]   |   Loss: {loss}   |   Accuracy: {acc}")

                # WEIGHTS UPDATE
                E = mb_y - mb_y_pred
                dw = self.lr * mb_X.T@E
                self.weights = self.weights + dw
                self.w0 += self.lr * np.sum(E)
            
            accs.append(acc)
            losses.append(loss)

        return losses, accs


class AdalineStochastic:

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

    def _shuffle(self, X, y):
        perm = np.random.permutation(len(y))
        return X[perm], y[perm]

    # Non implementa la riduzione graduale del tasso di apprendimento
    def fit(self, X, y, n_epochs=100):
        losses = []
        accs = [] # accuracy (%)
        n_samples = len(y)
        for n in range(n_epochs):
            epoch_loss = 0
            epoch_acc = 0
            for i in range(n_samples):
                # la riga i-esima corrisponde all'i-esimo sample del dataset, x^(i)
                x = X[i, :]
                y_pred = self.activate(x)

                # LOSS AND STATISTICS CALCULATION
                output = np.sign(y_pred)
                epoch_loss += self.loss(y, y_pred)
                epoch_acc += int(output == y[i])

                # WEIGHTS UPDATE
                E = y[i] - y_pred
                dw = self.lr * E * x
                self.weights = self.weights + dw
                self.w0 += self.lr * E

            epoch_loss /= n_samples
            epoch_acc /= n_samples
            print(f"[Epoch {n+1}] Loss: {epoch_loss}   |   Accuracy: {epoch_acc*100}")
            losses.append(epoch_loss)
            accs.append(epoch_acc*100)
            X, y = self._shuffle(X,y)
        return losses, accs