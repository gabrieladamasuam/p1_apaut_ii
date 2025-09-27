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
        """
        Inicializa el selector mRMR.

        Parámetros:
        - n_features: Número de características a seleccionar.
        """
        self.n_features = n_features
        self.selected_features_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Ajusta el selector mRMR a los datos.

        Parámetros:
        - X: Matriz de características de entrada.
        - y: Vector de etiquetas objetivo.
        """
        n_samples, n_features = X.shape
        selected = []
        not_selected = list(range(n_features))

        # Calcular relevancia (mutua información con y)
        relevance = mutual_info_regression(X, y)
        # Inicialmente, seleccionar la característica más relevante
        idx = np.argmax(relevance)
        selected.append(idx)
        not_selected.remove(idx)

        # Precalcular la matriz de redundancia (mutua información entre características)
        # Usamos mutual_info_regression para cada par (puede ser costoso)
        # Para evitar for, usamos broadcasting y np.apply_along_axis
        # Pero mutual_info_regression solo acepta 2D X y 1D y, así que usamos comprensión
        redundancy = np.zeros((n_features, n_features))
        for i in range(n_features):
            redundancy[i, :] = [mutual_info_regression(X[:, [i]], X[:, j])[0] for j in range(n_features)]

        # Selección iterativa
        for _ in range(self.n_features - 1):
            # Para cada no seleccionada, calcular score mRMR
            # score = relevancia - redundancia media con seleccionadas
            rel = relevance[not_selected]
            red = np.array([
                redundancy[j, selected].mean() if selected else 0.0
                for j in not_selected
            ])
            scores = rel - red
            idx_in_not_selected = np.argmax(scores)
            idx = not_selected[idx_in_not_selected]
            selected.append(idx)
            not_selected.remove(idx)

        self.selected_features_ = np.array(selected)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Disminuye la matriz de características a las características seleccionadas.     
        Parámetros:
        - X: Matriz de características de entrada.
        Retorna:
        - X_reduced: Matriz de características reducida.
        """
        if self.selected_features_ is None:
            raise RuntimeError("Debes llamar a fit antes de transform.")
        return X[:, self.selected_features_]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)