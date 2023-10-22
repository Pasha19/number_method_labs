import lab4
import numpy as np


def test_simple_iteration():
    matrix = np.array([
        [5, 1, 2],
        [1, 4, 1],
        [2, 1, 3],
    ], dtype="float64")
    lmbda, vector = lab4.simple_iteration(matrix)
    assert np.allclose(matrix @ vector, lmbda * vector)


def test_rotation():
    matrix = np.array([
        [5, 1, 2],
        [1, 4, 1],
        [2, 1, 3],
    ], dtype="float64")
    lambdas, vectors = lab4.rotation(matrix)
    for i in range(len(matrix)):
        lmbda = lambdas[i]
        vector = vectors[:, i]
        assert np.allclose(matrix @ vector, lmbda * vector)
