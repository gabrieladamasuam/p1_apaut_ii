"""
Planta de la práctica 1: mRMR (Minimum Redundancy Maximum Relevance)
Instrucciones:
- Completa la implementación de la clase mRMR.
- Asegúrate de que todos los tests en test_practica_1.py pasen correctamente. Para ello ejecuta `pytest` en la terminal.

Requisitos:
- Python 3.9+
- numpy, scipy (opcional), scikit-learn, pandas (opcional para reportes)

Autor: Eva Blázquez y Gabriela Damas
"""
import numpy as np
from typing import Optional, Dict, Tuple
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

class mRMR:
    """
    Selector de atributos mRMR (Minimum Redundancy, Maximum Relevance).
    - Relevancia: I(X_j; y) usando mutual_info_classif (clasificación).
    - Redundancia: media de I(X_j; X_s) con las características ya seleccionadas (regresión entre pares continuos).
    Compatible con sklearn (fit/transform, get_params/set_params).
    """
    def __init__(self, n_features: int, random_state: Optional[int] = 42):
        if not isinstance(n_features, int) or n_features < 1:
            raise ValueError("n_features debe ser un entero positivo")
        self.n_features = n_features
        self.random_state = random_state
        self.selected_features_: Optional[np.ndarray] = None

    def get_params(self, deep: bool = True) -> dict:
        return {"n_features": self.n_features, "random_state": self.random_state}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _pair_mi(self, X: np.ndarray, i: int, j: int, cache: Dict[Tuple[int, int], float]) -> float:
        """MI simétrica entre dos características continuas X[:, i] y X[:, j] con caché."""
        a, b = (i, j) if i <= j else (j, i)
        if (a, b) in cache:
            return cache[(a, b)]
        mi = float(mutual_info_regression(X[:, [a]], X[:, b], random_state=self.random_state)[0])
        cache[(a, b)] = mi
        return mi

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X debe ser 2D (n_samples, n_features)")
        n_samples, n_features = X.shape
        if n_features == 0:
            raise ValueError("X no contiene características")
        k = min(self.n_features, n_features)

        # Relevancia: MI de cada feature con y (clasificación)
        relevance = mutual_info_classif(X, y, random_state=self.random_state)

        # Selección greedy
        selected = []
        remaining = list(range(n_features))
        cache_mi_feat: Dict[Tuple[int, int], float] = {}

        # 1) arranca con la más relevante
        first = int(np.argmax(relevance))
        selected.append(first)
        remaining.remove(first)

        # 2) iterativo: maximiza relevancia - redundancia media
        while len(selected) < k and remaining:
            scores = []
            for j in remaining:
                if selected:
                    reds = [self._pair_mi(X, j, s, cache_mi_feat) for s in selected]
                    red_mean = float(np.mean(reds)) if len(reds) else 0.0
                else:
                    red_mean = 0.0
                scores.append(relevance[j] - red_mean)
            j_best = remaining[int(np.argmax(scores))]
            selected.append(j_best)
            remaining.remove(j_best)

        self.selected_features_ = np.array(selected, dtype=int)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.selected_features_ is None:
            raise RuntimeError("Debes llamar a fit antes de transform.")
        X = np.asarray(X)
        return X[:, self.selected_features_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)