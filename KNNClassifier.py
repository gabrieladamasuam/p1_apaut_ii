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

Autor: 
"""

import numpy as np
import scipy as sp
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
            self.distance_metric = self.euclidean_distance
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
        self.X_train = X
        self.y_train = y


    def predict(self, X):
        X = np.array(X)
        y_pred = []
        for x in X:
            # Calcular distancias a todos los puntos de entrenamiento
            distances = np.linalg.norm(self.X_train - x, axis=1)
            # Obtener los índices de los k vecinos más cercanos
            neighbors_idx = np.argsort(distances)[:self.k]
            # Obtener las etiquetas de los vecinos
            neighbor_labels = self.y_train[neighbors_idx]
            # Voto mayoritario
            values, counts = np.unique(neighbor_labels, return_counts=True)
            majority_label = values[np.argmax(counts)]
            y_pred.append(majority_label)
        return np.array(y_pred)
    
    @staticmethod
    def euclidean_distance(x1, x2):
        """Calcula la distancia euclidiana entre dos puntos.

        Args:
            x1 (np.ndarray): Primer punto.
            x2 (np.ndarray): Segundo punto.

        Returns:
            float: Distancia euclidiana entre x1 y x2.
        """
        return np.sqrt(np.sum((x1 - x2)**2))