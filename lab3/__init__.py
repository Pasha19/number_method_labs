import numpy as np
import numpy.typing as npt


class MatrixSizeException(Exception):
    pass


def validate_slae_matrix(matrix: npt.NDArray) -> None:
    shape = matrix.shape
    if len(shape) != 2 or shape[0] + 1 != shape[1]:
        raise MatrixSizeException()


def reverse(matrix: npt.NDArray) -> npt.NDArray:
    shape = matrix.shape
    result = np.zeros((shape[0],))
    for i in range(shape[0] - 1, -1, -1):
        result[i] = matrix[i][shape[0]]
        for j in range(shape[0] - 1, i, -1):
            result[i] -= result[j] * matrix[i][j]
    return result


def method_gauss(matrix: npt.NDArray) -> npt.NDArray:
    shape = matrix.shape
    validate_slae_matrix(matrix)
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
    validate_slae_matrix(matrix)
    for k in range(shape[0]):
        helper_rect(matrix, k)
    return reverse(matrix)


def method_gauss_max_element(matrix: npt.NDArray) -> npt.NDArray:
    shape = matrix.shape
    validate_slae_matrix(matrix)
    for k in range(shape[0]):
        max_ind = abs(matrix[k:, k]).argmax() + k
        if k != max_ind:
            matrix[[k, max_ind]] = matrix[[max_ind, k]]
        helper_rect(matrix, k)
    return reverse(matrix)


def helper_normilize(matrix: npt.NDArray) -> None:
    shape = matrix.shape
    for i in range(shape[0]):
        max_ind = abs(matrix[i, :-1]).argmax()
        if i != max_ind:
            matrix[[i, max_ind]] = matrix[[max_ind, i]]
    for i in range(shape[0]):
        matrix[i] /= -matrix[i][i]
        matrix[i][i] = 0.0
        matrix[i][-1] *= -1.0


def method_simple_iteration(
        matrix: npt.NDArray,
        eps: float = 1.0e-6,
        max_iterations: int = 20,
        norm=lambda x: abs(x).max()
):
    shape = matrix.shape
    validate_slae_matrix(matrix)
    helper_normilize(matrix)
    x0 = np.copy(matrix[:, shape[0]])
    for i in range(max_iterations):
        x = matrix[:, :-1].dot(x0) + matrix[:, -1]
        if norm(x - x0) < eps:
            break
        x0 = x
    return x0


def method_seidel(
        matrix: npt.NDArray,
        eps: float = 1.0e-6,
        max_iterations: int = 20,
        norm=lambda x: abs(x).max()
):
    shape = matrix.shape
    validate_slae_matrix(matrix)
    helper_normilize(matrix)
    x0 = np.zeros((shape[0],))
    for k in range(max_iterations):
        x = np.zeros((shape[0],))
        for i in range(shape[0]):
            for j in range(i):
                x[i] += matrix[i][j] * x[j]
            for j in range(i, shape[0]):
                x[i] += matrix[i][j] * x0[j]
            x[i] += matrix[i][-1]
        if norm(x - x0) < eps:
            break
        x0 = x
    return x0
