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
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Ajusta el selector mRMR a los datos.

        Parámetros:
        - X: Matriz de características de entrada.
        - y: Vector de etiquetas objetivo.
        """
        pass
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Disminuye la matriz de características a las características seleccionadas.     
        Parámetros:
        - X: Matriz de características de entrada.
        Retorna:
        - X_reduced: Matriz de características reducida.
        """
        pass
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass