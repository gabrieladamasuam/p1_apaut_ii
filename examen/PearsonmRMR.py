import numpy as np

class PearsonmRMR:
    def __init__(self, n_features: int, random_state: int = 42):
        if not isinstance(n_features, int) or n_features < 1:
            raise ValueError("n_features debe ser un entero positivo")
        self.n_features = n_features
        self.random_state = random_state
        self.selected_idx_ = None  # Se almacenarán los índices seleccionados

    def fit(self, X: np.ndarray, y: np.ndarray):
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

        # Paso 1: seleccionar el más relevante
        relevance = np.corrcoef(np.c_[X, y], rowvar=False)[:-1, -1]
        first = int(np.argmax(relevance))
        selected.append(first)
        remaining.remove(first)

        # Paso 2: iterar maximizando relevancia - redundancia media
        while len(selected) < min(self.n_features, n_cols) and remaining:
            scores = []
            for i in remaining:
                reds = [self._pair_corr(X, i, s, pairs) for s in selected] if selected else []
                red_mean = float(np.mean(reds)) if reds else 0.0
                scores.append(relevance[i] - red_mean)

            i_best = remaining[int(np.argmax(scores))]
            selected.append(i_best)
            remaining.remove(i_best)

        self.selected_idx_ = np.array(selected, dtype=int)
        return self

    def _pair_corr(self, X: np.ndarray, i: int, j: int, pairs: dict) -> float:
        if (i, j) in pairs:
            return pairs[(i, j)]
        if (j, i) in pairs:
            return pairs[(j, i)]
        c = np.corrcoef(X[:, i], X[:, j])[0, 1]
        pairs[(i, j)] = c
        return c

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.selected_idx_ is None:
            raise RuntimeError("Debes llamar a fit antes de transform.")
        X = np.asarray(X)
        return X[:, self.selected_idx_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    def get_params(self, deep: bool = True) -> dict:
        return {"n_features": self.n_features, "random_state": self.random_state}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self