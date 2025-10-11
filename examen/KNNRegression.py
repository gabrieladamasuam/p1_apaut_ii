import numpy as np
from scipy.spatial import distance

class KNNRegressor:
    def __init__(self, k=3, distance_metric=None, p=2):
        if not isinstance(k, int) or k < 1:
            raise ValueError("k debe ser un entero positivo")
        self.k = k
        self.p = p
        self.distance_metric = distance_metric or 'minkowski'
        self.X_train = None
        self.y_train = None
        self._fitted = False

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X e y deben tener el mismo número de ejemplos")
        if self.k > X.shape[0]:
            raise ValueError("k no puede ser mayor que el número de muestras de entrenamiento")
        self.X_train = X
        self.y_train = y
        self._fitted = True
        return self

    def _distance_matrix(self, X):
        metric = self.distance_metric
        if callable(metric):
            return distance.cdist(X, self.X_train, metric=metric)
        if metric == 'minkowski':
            return distance.cdist(X, self.X_train, metric='minkowski', p=self.p)
        # otras métricas de scipy (euclidean, cityblock, etc.)
        return distance.cdist(X, self.X_train, metric=metric)

    def predict(self, X):
        if not self._fitted:
            raise RuntimeError("Debes llamar a fit antes de predecir")
        X = np.asarray(X)
        # Si X es un único vector (d,), convertirlo a (1, d) para operar como lote de 1 muestra
        if X.ndim == 1:
            X = X.reshape(1, -1)
        dists = self._distance_matrix(X)
        # Selecciona los k menores sin ordenar el bloque (O(n) por fila), suficiente para voto mayoritario y más eficiente que argsort (O(n log n))
        idx = np.argpartition(dists, kth=self.k - 1, axis=1)[:, :self.k]
        neigh_y = self.y_train[idx]
        pred = neigh_y.mean(axis=1)
        return pred.astype(float)

    def get_params(self, deep=True):
        return {"k": self.k, "distance_metric": self.distance_metric, "p": self.p}

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self