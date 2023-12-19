import numpy as np

class Ovr:

    def __init__(self):
        pass
    
    def fit(self, X, y, classifiers: list, n_epochs=10, verbose=True):
        clf_accuracies = []
        self.classifiers = classifiers
        for i, classifier in enumerate(classifiers):
            classifier_y = np.where(y == i, 1, -1)
            if verbose:
                print(f"Allenando il classificatore #{i+1}")
            accs = classifier.fit(X, classifier_y, n_epochs=n_epochs, verbose=verbose)
            clf_accuracies.append(accs)

        # indici de classificatori in base alla loro accuratezza in ordine decrescente
        self.sorted_indices = np.argsort([clf_accuracy[-1] for clf_accuracy in clf_accuracies])[::-1]
        return clf_accuracies

    def predict(self, X):
        if len(X.shape) > 1:
            y_pred = np.zeros(X.shape[0])
            # per ogni sample
            for n_sample, x in enumerate(X):
                # in ordine decrescente in base alla loro accuratezza controllo se uno dei classificatori
                # riconosce il sample come proprio
                for classifier_idx in self.sorted_indices:
                    if self.classifiers[classifier_idx].predict(x) == 1:
                        y_pred[n_sample] = classifier_idx
                        break
            return y_pred
        else:
            for classifier_idx in self.sorted_indices:
                if self.classifiers[classifier_idx].predict(x) == 1:
                    return classifier_idx
                
    def get_accuracy(self, y, y_pred):
        n_errors = np.count_nonzero(y-y_pred)
        n_samples = len(y)
        return 100*(n_samples - n_errors)/n_samples