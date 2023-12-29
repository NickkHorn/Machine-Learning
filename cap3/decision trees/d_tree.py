import numpy as np

'''
Dato un vettore che specifica la distribuzione dei target nei rami di un nodo 
ritorna l'entropia di Shannon per misurare l'impurità del criterio utilizzato
'''
def entropy(subs_class_dist):
    # divido per il numero di classi per ottenere la frequenza di una classe rispetto all'insieme
    frequency_probs = subs_class_dist/np.sum(subs_class_dist)
    return -np.sum(frequency_probs*np.log2(frequency_probs))

'''
Dato un vettore che specifica la distribuzione dei target nei rami di un nodo 
ritorna l'impurità di Gini per misurare l'impurità del criterio utilizzato
'''
def gini_impurity(subs_class_dist):
    frequency_probs = subs_class_dist/np.sum(subs_class_dist)
    return 1 - np.sum(frequency_probs**2)

class DecisionTree:

    def __init__(self, metric='entropy'):
        # ogni criterio è un tuple (f, t), che applicato risulta nella suddivisione binaria del nodo
        # xf >= t
        self.unique_labels, self.labels_counts = None, None
        self.metric = metric
        self.impurity = None
        self.child_a, self.child_b = None, None

    '''
        Da chiamare solo per creare nuovi sottonodi
    '''
    @classmethod
    def fromdata(cls, y, metric='entropy'):
        cls.unique_labels, cls.label_counts = cls.__get_labels_count(cls, y)
        
        if metric == 'entropy':
            cls.impurity = entropy(cls.label_counts)
        elif metric == 'gini':
            cls.impurity = gini_impurity(cls.label_counts)
        else:
            raise TypeError("Metrica di impurità invalida!")

        return cls(metric)
    

    def generate_target_data(self, y, metric='entropy'):
        self.unique_labels, self.label_counts = self.__get_labels_count(y)
        
        if metric == 'entropy':
            self.impurity = entropy(self.label_counts)
        elif metric == 'gini':
            self.impurity = gini_impurity(self.label_counts)
        else:
            raise TypeError("Metrica di impurità invalida!")


    '''
        Dal vettore dei target ritorna due valori:
            1. Un vettore con tutte le classi uniche nella classificazione
            2. Un vettore dove per ciascun indice del primo corrisponde la frequenza di quel target
    '''
    def __get_labels_count(self, labels):
        unique_labels = [] # ottengo tutti gli indici/target delle classi uniche
        label_counts = [] # per ogni indice in unique_label questa lista contiene il numero di volte che appare in data
        for target in labels:
            if target not in unique_labels:
                unique_labels.append(target)
                label_counts.append(1) # se la classe è nuova allora per ora appare solo una volta
            else:
                labels_list_idx = unique_labels.index(target)
                label_counts[labels_list_idx] += 1

        return unique_labels, label_counts

    '''
    Ritorna il guadagno informativo ottenuto dati i due sottoinsiemi creati dalla suddivisione, i due parametri
    sono delle liste dove ogni elemento rappresenta il numero di volte che un sample della classe corrispondente
    a quell'indice appare nel ramo corrispondente del nodo
    '''
    def IG(self, subset_a_ccounts, subset_b_ccounts):
        # ammontare di sample che finiscono nel nodo A e nel nodo B
        Na_capacity = np.sum(subset_a_ccounts)
        Nb_capacity = np.sum(subset_b_ccounts)
        total_capacity = Na_capacity + Nb_capacity
        if self.metric == 'entropy':
            suba_impurity = entropy(subset_a_ccounts)
            subb_impurity = entropy(subset_b_ccounts)
        elif self.metric == 'gini':
            suba_impurity = gini_impurity(subset_a_ccounts)
            subb_impurity = gini_impurity(subset_b_ccounts)

        return self.impurity - suba_impurity*Na_capacity/total_capacity - subb_impurity*Nb_capacity/total_capacity

    '''
        Ritorna una lista di tuple (f, t) che cambierebbero la decisione posta sui dati di addestramento, non
        serve considerare il caso in cui t >= x[f] per ogni x in input, in quanto se quella fosse la migliore 
        classificazione allora il guadagno informativo migliore sarebbe 0, e dunque il nodo non può imparare nulla
    '''
    def __get_features_to_check(self, X):
        # per ogni sample n registro il potenziale t per la feature f
        features_raw = [(f, X[n,f]) for n in range(X.shape[0]) for f in range(X.shape[1])]
        features = list(dict.fromkeys(features_raw)) # rimuovi i duplicati
        return features

    '''
        Trova e salva nella classe il miglior criterio in termini di guadagno informativo, presi come parametri
        la matrice di input, il vettore dei targets e una lista delle coppie (f, t) dei criteri validi, la funzione
        ritorna 
            1 - La coppia (f, t) del criterio con il miglior guadagno informativo
            2 - il valore del guadagno informativo ottenuto dalla suddivisione in base al criterio scelto
            3 - gli indici delle colonne in X e dei targets in y che sono classificati nel primo nodo figlio
            4 - gli indici delle colonne in X e dei targets in y che sono classificati nel primo secondo figlio
    '''
    def __apply_best_criterion(self, X, y, valid_criteria):
        bc = valid_criteria[0] # best criterion, il miglior criterio
        bc_ig = -1e5 # best criterion information gain, il guadagno informativo del migliore criterio disponibile
        bc_suba_indices, bc_subb_indices = None, None
    
        # Trova il migliore criterio
        for criterion in valid_criteria:
            f, t = criterion
            suba_indices = np.where(X[:, f] >= t)[0] # indice degli elementi che ricadrebbero nel sottoinsieme A
            subb_indices = np.where(X[:, f] < t)[0] # indice degli elementi che ricadrebbero nel sottoinsieme B

            suba_labels = y[suba_indices] # target che ricadono in A
            subb_labels = y[subb_indices] # target che ricadono in B
            
            unqiue_a_labels, subset_a_ccounts = self.__get_labels_count(suba_labels)
            unqiue_b_labels, subset_b_ccounts = self.__get_labels_count(subb_labels)

            criterion_ig = self.IG(subset_a_ccounts, subset_b_ccounts)

            if criterion_ig > bc_ig:
                bc_ig = criterion_ig
                bc = criterion
                bc_suba_indices = suba_indices
                bc_subb_indices = subb_indices

        return bc, bc_ig, bc_suba_indices, bc_subb_indices

    '''
        Metodo fit da chiamare SOLO ESTERNAMENTE nel nodo genitore, che richiede quindi
        i dati di addestramento, i nodi figlio avranno un metodo a sè in quanto i dati 
        gli saranno passati alla creazione tramite il classmethod "fromdata"
    '''
    def fit(self, X, y, tree_depth=4):
        # inizializza i dati se non sono già stati caricati, cosa che avviene se il nodo è la testa dell'albero
        if self.unique_labels is None or self.labels_counts is None or self.impurity is None:
            self.generate_target_data(y, metric = self.metric)

        # base case
        if tree_depth == 0:
            return

        valid_criteria = self.__get_features_to_check(X)
        self.bc, self.bc_ig, bc_suba_indices, bc_subb_indices = self.__apply_best_criterion(X, y, valid_criteria)

        A_X, B_X = X[bc_suba_indices, :], X[bc_subb_indices, :]        
        A_y, B_y = y[bc_suba_indices], y[bc_subb_indices]

        # passa alla prossima iterazione
        self.child_a, self.child_b = DecisionTree.fromdata(A_y), DecisionTree.fromdata(B_y)
        self.child_a.fit(A_X, A_y, tree_depth=tree_depth-1)
        self.child_b.fit(B_X, B_y, tree_depth=tree_depth-1)

    '''
        Predice la classe di un singolo sample dopo che l'albero è stato creato
    '''
    def _predict_single(self, x):
        # base case -> ritorna la classe più frequente nel nodo
        if self.child_a is None or self.child_b is None:
            most_freq_class_idx = np.argmax(self.labels_counts)
            return self.unique_labels[most_freq_class_idx]

        f, t = self.bc
        return self.child_a._predict_single(x) if x[f] >= t else self.child_b._predict_single(x)


if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.tree import DecisionTreeClassifier

    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y)

    tree = DecisionTreeClassifier(criterion='entropy', 
                                max_depth=2, 
                                random_state=1)
    tree = DecisionTree(metric='entropy')
    tree.fit(X_train, y_train, tree_depth=3)
    print(f"Nodo A criterio: x[{tree.bc[0]}] >= {tree.bc[1]}")
    
    print(f"Nodo B1 criterio: x[{tree.child_a.bc[0]}] >= {tree.child_a.bc[1]} ")
    print(f"Nodo B2 criterio: x[{tree.child_b.bc[0]}] >= {tree.child_b.bc[1]} ")

    print(f"Nodo C1 criterio: x[{tree.child_a.child_a.bc[0]}] >= {tree.child_a.child_a.bc[1]} ")
    print(f"Nodo C2 criterio: x[{tree.child_a.child_b.bc[0]}] >= {tree.child_a.child_b.bc[1]} ")
    print(f"Nodo C3 criterio: x[{tree.child_b.child_a.bc[0]}] >= {tree.child_b.child_a.bc[1]} ")
    print(f"Nodo C4 criterio: x[{tree.child_b.child_b.bc[0]}] >= {tree.child_b.child_b.bc[1]} ")
