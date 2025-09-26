# test_knnclassifier.py
import numpy as np
from KNNClassifier import KNNClassifier
from mRMR import mRMR


def make_toy_dataset():
    # Dataset binario linealmente separable muy simple
    X = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [10.0, 10.0],
        [9.0, 9.0],
        [10.0, 9.0],
        [9.0, 10.0],
    ])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return X, y
# Distancia absoluta
def manhattan(a, b):
    return np.sum(np.abs(a - b))

def test_knn_fit_and_predict_uniform():
    X, y = make_toy_dataset()
    clf = KNNClassifier(k=3, distance_metric=manhattan)
    clf.fit(X, y)

    # Un punto cercano al grupo clase 0
    x_test = np.array([[0.2, 0.1]])
    y_pred = clf.predict(x_test)
    assert y_pred[0] == 0

    # Un punto cercano al grupo clase 1
    x_test = np.array([[9.5, 11.0]])
    y_pred = clf.predict(x_test)
    assert y_pred[0] == 1


data_mrmr = (np.array([
    [0, 2, 1],
    [1, 5, 0],
    [1, 8, 0],
    [0, 0, 1],
    [0, 1, 1]
]), np.array([0, 1, 1, 0, 0]))

def test_mRMR_basic(data_mrmr):
    """
    Test para verificar si el método mRMR selecciona correctamente los atributos.
    """
    X,y = data_mrmr
    my_mRMR =  mRMR(2)
    my_mRMR.fit(X, y)
    X_transformed = my_mRMR.transform(X)
    assert X_transformed.shape[1] == 2
    assert X_transformed == X[:, [0, 2]]  
    
