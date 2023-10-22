import numpy as np
import numpy.typing as npt
from lab3 import MatrixSizeException
from math import atan2, cos, sin


def simple_iteration(matrix: npt.NDArray, eps: float = 1.0e-6, max_iterations: int = 100) -> tuple[float, npt.NDArray]:
    size = matrix.shape[0]
    if size != matrix.shape[1]:
        raise MatrixSizeException()
    x0 = np.ones(size)
    x = np.dot(matrix, x0)
    lambda0 = x[0] / x0[0]
    x0 = x.copy()
    for k in range(max_iterations):
        max_abs = np.max(abs(x0))
        if max_abs > 1000.0:
            x0 /= max_abs
        x = np.dot(matrix, x0)
        lambda1 = np.mean(x / x0)
        x0 = x
        if abs(lambda1 - lambda0) < eps:
            break
        lambda0 = lambda1
    return lambda0, x0 / max(abs(x0))


def max_top_element_index(matrix: npt.NDArray) -> tuple[int, int]:
    size = matrix.shape[0]
    i0, j0 = 0, 1
    for i in range(size):
        for j in range(i + 1, size):
            if abs(matrix[i, j]) > abs(matrix[i0, j0]):
                i0, j0 = i, j
    return i0, j0


def rotation(matrix: npt.NDArray, eps: float = 1.0e-6, max_iterations: int = 100) -> tuple[npt.NDArray, npt.NDArray]:
    size = matrix.shape[0]
    matrix_copy = np.copy(matrix)
    result = np.identity(size, dtype="float64")
    if size != matrix_copy.shape[1]:
        raise MatrixSizeException()
    for k in range(max_iterations):
        i, j = max_top_element_index(matrix_copy)
        a = matrix_copy[i, j]
        if abs(a) < eps:
            break
        a_ii = matrix_copy[i, i]
        a_jj = matrix_copy[j, j]
        phi = 0.5 * atan2(2 * a, a_ii - a_jj)
        h = np.identity(size, dtype="float64")
        h[i, i] = cos(phi)
        h[i, j] = -sin(phi)
        h[j, i] = sin(phi)
        h[j, j] = cos(phi)
        result @= h
        matrix_copy = h.T @ matrix_copy @ h
    return np.diagonal(matrix_copy), result
