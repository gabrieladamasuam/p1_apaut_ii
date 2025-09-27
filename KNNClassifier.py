import numpy as np
from scipy.stats import mode
from scipy.spatial import distance

class KNNClassifier:
    def __init__(self, k=3, distance_metric=None, p=2):
        self.k = int(k)
        self.p = p
        self.distance_metric = 'minkowski' if distance_metric is None else distance_metric
        self.X_train = None
        self.y_train = None
        self._fitted = False

    def fit(self, X, y):
        X = np.asarray(X); y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X e y deben tener el mismo número de ejemplos")
        if self.k < 1:
            raise ValueError("k debe ser >= 1")
        if self.k > X.shape[0]:
            raise ValueError("k no puede ser mayor que el número de muestras de entrenamiento")
        self.X_train = X
        self.y_train = y
        self._fitted = True
        return self

    def _pairwise_distances(self, X):
        metric = self.distance_metric
        if callable(metric):
            return distance.cdist(X, self.X_train, metric=metric)
        if metric is None or metric == 'minkowski':
            return distance.cdist(X, self.X_train, metric='minkowski', p=self.p)
        return distance.cdist(X, self.X_train, metric=metric)

    def predict(self, X):
        if not self._fitted:
            raise RuntimeError("Debes llamar a fit antes de predict.")
        X = np.asarray(X)
        dists = self._pairwise_distances(X)
        idx = np.argpartition(dists, kth=self.k - 1, axis=1)[:, :self.k]
        neighbor_labels = self.y_train[idx]
        m = mode(neighbor_labels, axis=1, keepdims=False)
        majority = getattr(m, "mode", m)
        return np.asarray(majority).ravel().astype(self.y_train.dtype)

    def score(self, X, y):
        X = np.asarray(X); y = np.asarray(y)
        return np.mean(self.predict(X) == y)

    def get_params(self, deep=True):
        return {"k": self.k, "distance_metric": self.distance_metric, "p": self.p}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self