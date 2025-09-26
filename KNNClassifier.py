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

Autor: Eva Blazquez y Gabriela Damas
"""

import numpy as np
from scipy.stats import mode
from scipy.spatial import distance
import sklearn as sk
import pandas as pd


class KNNClassifier:
    def __init__(self, k=3, distance_metric=None):
        """
        Inicializa el clasificador KNN.

        Parámetros:
        - k: Número de vecinos a considerar.
        - distance_metric: Función de distancia que toma dos vectores y devuelve un escalar.
        """
        self.k = k
        if distance_metric is None:
            self.distance_metric = distance.euclidean   # usa scipy
        else:
            self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Ajusta el modelo KNN a los datos de entrenamiento.

        Parámetros:
        - X: Matriz de características de entrenamiento.
        - y: Vector de etiquetas de entrenamiento.
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        """
        Realiza predicciones para las muestras de entrada.

        Parámetros:
        - X: Matriz de características de las muestras a predecir.

        Retorna:
        - np.ndarray: Vector de etiquetas predichas.
        """
        X = np.array(X)
        # Matriz de distancias entre cada muestra de X y cada muestra de entrenamiento
        dists = np.linalg.norm(self.X_train[None, :, :] - X[:, None, :], axis=2)
        # Índices de los k vecinos más cercanos
        neighbors_idx = np.argsort(dists, axis=1)[:, :self.k]
        # Etiquetas de los vecinos
        neighbor_labels = self.y_train[neighbors_idx]
        # Voto mayoritario por fila
        majority_labels, _ = mode(neighbor_labels, axis=1)
        return majority_labels.ravel()

