import lab3
import numpy as np


def test_method_gauss():
    slae = np.array([
        [2, 1, 4, 16],
        [3, 2, 1, 10],
        [1, 3, 3, 16],
    ], dtype="float64")
    solution = lab3.method_gauss(np.copy(slae))
    shape = slae.shape
    matr = slae[:shape[0], :shape[0]]
    col = slae[:, [shape[0]]]
    assert np.allclose(matr.dot(solution), col)
