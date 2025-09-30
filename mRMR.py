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
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import numpy as np

class mRMR:
    def __init__(self, n_features: int, random_state = 42):
        """Inicializa el selector mRMR.

        Parámetros:
        - n_features: Número de características a seleccionar.
        - random_state: Semilla para reproducibilidad.
        """
        if not isinstance(n_features, int) or n_features < 1:
            raise ValueError("n_features debe ser un entero positivo")
        self.n_features = n_features
        self.random_state = random_state
        self.selected_features_ = None  # Se almacenarán los índices seleccionados

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Ajusta el selector mRMR a los datos.

        Parámetros:
        - X: Matriz de características de entrada.
        - y: Vector de etiquetas objetivo.
        """
        # Convierte X e y a arrays de numpy
        X = np.asarray(X)
        y = np.asarray(y)

        # Validación de dimensiones
        if X.ndim != 2:
            raise ValueError("X debe ser 2D")

        n_cols = X.shape[1]
        if n_cols == 0:
            raise ValueError("X no contiene características")

        # Inicialización de listas para selección de características
        selected = []
        remaining = list(range(n_cols))
        pairs = {}

        # Paso 1: seleccionar el atributo más relevante
        relevance = mutual_info_classif(X, y, random_state=self.random_state) # Relevancia: Información mutua entre cada atributo y la variable objetivo
        first = int(np.argmax(relevance))
        selected.append(first)
        remaining.remove(first)

        # Paso 2: seleccionar atributos que maximizan relevancia - redundancia media
        while len(selected) < min(self.n_features, n_cols) and remaining:
            scores = []
            for i in remaining:
                # Redundancia: media de la información mutua entre i y los atributos ya seleccionados
                if selected:
                    reds = [self._pair_im(X, i, s, pairs) for s in selected]
                    red_mean = float(np.mean(reds)) if len(reds) else 0.0
                else:
                    red_mean = 0.0
                scores.append(relevance[i] - red_mean)
           
            # Seleccionar el atributo con mejor puntuación
            i_best = remaining[int(np.argmax(scores))]
            selected.append(i_best)
            remaining.remove(i_best)

        self.selected_features_ = np.array(selected, dtype=int)
        return self

    def _pair_im(self, X: np.ndarray, i: int, j: int, pairs: dict) -> float:
        """
        Calcula la información mutua entre dos características, utilizando caché para evitar cálculos repetidos.
        
        Parámetros:
        - X: Matriz de características.
        - i, j: Índices de las características a comparar.
        - cache: Diccionario para almacenar resultados ya calculados.
        
        Retorna:
        - mi: Información mutua entre las características i y j.
        """
        if (i, j) in pairs:
            return pairs[(i, j)]
        elif (j, i) in pairs:
            return pairs[(j, i)]
        mi = mutual_info_regression(X[:, [i]], X[:, j], random_state=self.random_state)[0]
        pairs[(i, j)] = mi
        return mi

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
        X = np.asarray(X)
        return X[:, self.selected_features_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Ajusta el modelo y transforma los datos.
        
        Parámetros:
        - X: Matriz de características de entrada.
        - y: Vector de etiquetas objetivo.
        
        Retorna:
        - X_reduced: Matriz de características reducida.
        """
        self.fit(X, y)
        return self.transform(X)

    def get_params(self, deep: bool = True) -> dict:
        """
        Retorna los parámetros del modelo.
        
        Parámetros:
        - deep: Si es True, también retorna los parámetros de los sub-objetos.
       
         Retorna:
        - params: Diccionario con los parámetros del modelo.
        """
        return {"n_features": self.n_features, "random_state": self.random_state}

    def set_params(self, **params):
        """
        Establece los parámetros del modelo.
        
        Parámetros:
        - params: Diccionario con los parámetros a establecer.
        
        Retorna:
        - self: El objeto con los parámetros actualizados.
        """
        for k, v in params.items():
            setattr(self, k, v)
        return self