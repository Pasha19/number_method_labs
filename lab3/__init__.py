import numpy as np
import numpy.typing as npt


class MatrixSizeException(Exception):
    pass


def reverse(matr: npt.NDArray) -> npt.NDArray:
    shape = matr.shape
    result = np.zeros((shape[0], 1))
    for i in range(shape[0] - 1, -1, -1):
        result[i] = matr[i][shape[0]]
        for j in range(shape[0] - 1, i, -1):
            result[i] -= result[j] * matr[i][j]
    return result


def method_gauss(matrix: npt.NDArray) -> npt.NDArray:
    shape = matrix.shape
    if len(shape) != 2 or shape[0] + 1 != shape[1]:
        raise MatrixSizeException()

    for i in range(shape[0]):
        matrix[i] /= matrix[i][i]
        for j in range(i + 1, shape[0]):
            matrix[j] -= matrix[j][i] * matrix[i]

    return reverse(matrix)


def helper_rect(matrix: npt.NDArray, k: int) -> None:
    shape = matrix.shape
    for i in range(k + 1, shape[0]):
        for j in range(k + 1, shape[1]):
            matrix[i][j] -= matrix[k][j] * matrix[i][k] / matrix[k][k]
    matrix[k] /= matrix[k][k]
    matrix[k + 1:, k] = 0.0


def method_exclude_rect(matrix: npt.NDArray) -> npt.NDArray:
    shape = matrix.shape
    if len(shape) != 2 or shape[0] + 1 != shape[1]:
        raise MatrixSizeException()

    for k in range(shape[0]):
        helper_rect(matrix, k)

    return reverse(matrix)


def method_gauss_max_element(matrix: npt.NDArray) -> npt.NDArray:
    shape = matrix.shape
    if len(shape) != 2 or shape[0] + 1 != shape[1]:
        raise MatrixSizeException()

    for k in range(shape[0]):
        max_ind = abs(matrix[k:, k]).argmax() + k
        if k != max_ind:
            matrix[[k, max_ind]] = matrix[[max_ind, k]]
        helper_rect(matrix, k)

    return reverse(matrix)
