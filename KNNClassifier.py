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
        """Clasificador KNN.

        Args:
            k (int, optional): Número de vecinos a considerar. Defaults to 3.
            distance_metric (str o callable, optional): Métrica de distancia a utilizar. Defaults to None.
            p (int, optional): Parámetro p para la distancia de Minkowski. Defaults to 2.

        Raises:
            ValueError: Si los parámetros son inválidos.
        """
        if not isinstance(k, int) or k < 1:
            raise ValueError("k debe ser un entero positivo")
        self.k = k
        self.p = p
        self.distance_metric = distance_metric or 'minkowski'
        self.X_train = None
        self.y_train = None
        self._fitted = False

    def fit(self, X, y):
        """Entrena el clasificador KNN.

        Args:
            X (np.ndarray): Matriz de características de entrenamiento.
            y (np.ndarray): Vector de etiquetas de entrenamiento.

        Raises:
            ValueError: Si los datos de entrada son inválidos.
            ValueError: Si los datos de entrada son inválidos.

        Returns:
            KNNClassifier: El clasificador KNN entrenado.
        """
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
        """Calcula la matriz de distancias entre las muestras de X y las de entrenamiento.

        Args:
            X (np.ndarray): Matriz de características de las muestras a predecir.

        Returns:
            np.ndarray: Matriz de distancias (m, n) donde D[i, j] = dist(X[i], X_train[j]).
        """
        metric = self.distance_metric
        if callable(metric):
            return distance.cdist(X, self.X_train, metric=metric)
        if metric == 'minkowski':
            return distance.cdist(X, self.X_train, metric='minkowski', p=self.p)
        # otras métricas de scipy (euclidean, cityblock, etc.)
        return distance.cdist(X, self.X_train, metric=metric)

    def predict(self, X):
        """Realiza predicciones para las muestras de entrada.

        Args:
            X (np.ndarray): Matriz de características de las muestras a predecir.

        Raises:
            RuntimeError: Si el clasificador no ha sido entrenado.

        Returns:
            np.ndarray: Vector de etiquetas predichas para las muestras de entrada.
        """
        if not self._fitted:
            raise RuntimeError("Debes llamar a fit antes de predecir")
        X = np.asarray(X)
        # Si X es un único vector (d,), convertirlo a (1, d) para operar como lote de 1 muestra
        if X.ndim == 1:
            X = X.reshape(1, -1)
        dists = self._distance_matrix(X)
        # Selecciona los k menores sin ordenar el bloque (O(n) por fila), suficiente para voto mayoritario y más eficiente que argsort (O(n log n))
        idx = np.argpartition(dists, kth=self.k - 1, axis=1)[:, :self.k]
        vecinos = self.y_train[idx]
        m = mode(vecinos, axis=1, keepdims=False)
        pred = getattr(m, "mode", m)  # usa m.mode si existe; si no, m ya es el array del modo
        return np.asarray(pred).ravel().astype(self.y_train.dtype)

    def get_params(self, deep=True):
        """Obtiene los parámetros del clasificador KNN.

        Args:
            deep (bool, optional): Si se deben obtener también los parámetros de los subobjetos. Defaults to True.

        Returns:
            dict: Diccionario con los parámetros del clasificador.
        """
        return {"k": self.k, "distance_metric": self.distance_metric, "p": self.p}

    def set_params(self, **params):
        """Establece los parámetros del clasificador KNN.

        Returns:
            KNNClassifier: El clasificador KNN con los nuevos parámetros.
        """
        for param, value in params.items():
            setattr(self, param, value)
        return self