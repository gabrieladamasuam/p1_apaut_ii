#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plantilla de práctica: KNN, optimización de hiperparámetros y selección de atributos (mRMR)

Instrucciones:
- Completa la implementación de la clase KNNClassifier.
- Asegúrate de que todos los tests en test_practica_1.py pasen correctamente. Para ello ejecuta `pytest` en la terminal.

Requisitos:
- Python 3.9+
- numpy, scipy (opcional), scikit-learn, pandas (opcional para reportes)

Autor: Eva Blázquez y Gabriela Damas
"""

import numpy as np
from scipy.stats import mode
from scipy.spatial import distance

class KNNClassifier:
    def __init__(self, k=3, distance_metric=None, p=2):
        """
        Inicializa el clasificador KNN.

        Parámetros:
        - k: Número de vecinos a considerar.
        - distance_metric: Función de distancia que toma dos vectores y devuelve un escalar.
        """
        self.k = int(k)
        self.p = p
        self.distance_metric = 'minkowski' if distance_metric is None else distance_metric
        self.X_train = None
        self.y_train = None
        self._fitted = False

    def fit(self, X, y):
        """
        Ajusta el modelo KNN a los datos de entrenamiento.

        Parámetros:
        - X: Matriz de características de entrenamiento.
        - y: Vector de etiquetas de entrenamiento.
        """
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
        """
        Calcula la matriz de distancias entre los ejemplos en X y los ejemplos de entrenamiento.

        Args:
            X (np.ndarray): Matriz de características de los ejemplos a clasificar.

        Returns:
            np.ndarray: Matriz de distancias.
        """
        metric = self.distance_metric
        if callable(metric):
            return distance.cdist(X, self.X_train, metric=metric)
        if metric is None or metric == 'minkowski':
            return distance.cdist(X, self.X_train, metric='minkowski', p=self.p)
        return distance.cdist(X, self.X_train, metric=metric)

    def predict(self, X):
        """
        Realiza predicciones para los ejemplos en X.

        Args:
            X (np.ndarray): Matriz de características de los ejemplos a clasificar.

        Returns:
            np.ndarray: Vector de etiquetas predichas.
        """
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
        """
        Evalúa el rendimiento del modelo en los datos de prueba.

        Parámetros:
        - X: Matriz de características de prueba.
        - y: Vector de etiquetas de prueba.

        Returns:
            float: Precisión del modelo en los datos de prueba.
        """
        X = np.asarray(X); y = np.asarray(y)
        return np.mean(self.predict(X) == y)

    def get_params(self, deep=True):
        """
        Obtiene los parámetros del modelo.

        Args:
            deep (bool): Si se deben incluir los parámetros de los sub-modelos.

        Returns:
            dict: Diccionario con los parámetros del modelo.
        """
        return {"k": self.k, "distance_metric": self.distance_metric, "p": self.p}

    def set_params(self, **params):
        """ Establece los parámetros del modelo.

        Args:
            **params: Parámetros a establecer en el modelo.

        Returns:
            self
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self