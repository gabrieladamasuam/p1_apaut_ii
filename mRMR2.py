"""
Planta de la práctica 1: mRMR (Minimum Redundancy Maximum Relevance)
Instrucciones:
- Completa la implementación de la clase mRMR.
- Asegúrate de que todos los tests en test_practica_1.py pasen correctamente. Para ello ejecuta `pytest` en la terminal.

Requisitos:
- Python 3.9+
- numpy, scipy (opcional), scikit-learn, pandas (opcional para reportes)

Autor:
"""
from sklearn.feature_selection import mutual_info_regression
import numpy as np

class mRMR:
    def __init__(self, n_features: int):
        self.n_features = n_features
        self.selected_idx_ = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        relevance = mutual_info_regression(X, y)
        selected = []
        candidate_idx = list(range(X.shape[1]))

        for _ in range(self.n_features):
            scores = []
            for i in candidate_idx:
                if len(selected) == 0:
                    redundancy = 0
                else:
                    redundancy = np.mean([abs(np.corrcoef(X[:, i], X[:, j])[0, 1]) 
                                          for j in selected])
                scores.append(relevance[i] - redundancy)

            best_idx = candidate_idx[np.argmax(scores)]
            selected.append(best_idx)
            candidate_idx.remove(best_idx)

        self.selected_idx_ = selected
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.selected_idx_ is None:
            raise ValueError("Debes llamar a fit antes de transform.")
        return X[:, self.selected_idx_]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)