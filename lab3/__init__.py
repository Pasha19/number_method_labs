import numpy as np
import numpy.typing as npt


class MatrixSizeException(Exception):
    pass


def method_gauss(matr: npt.NDArray) -> npt.NDArray:
    shape = matr.shape
    if len(shape) != 2 or shape[0] + 1 != shape[1]:
        raise MatrixSizeException()

    for i in range(shape[0]):
        matr[i] /= matr[i][i]
        for j in range(i + 1, shape[0]):
            matr[j] -= matr[j][i] * matr[i]

    result = np.zeros((shape[0], 1))
    for i in range(shape[0] - 1, -1, -1):
        result[i] = matr[i][shape[0]]
        for j in range(shape[0] - 1, i, -1):
            result[i] -= result[j] * matr[i][j]

    return result
