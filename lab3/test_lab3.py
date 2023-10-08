import pytest
import lab3
import numpy as np


def get_methods():
    yield lab3.method_gauss
    yield lab3.method_exclude_rect


@pytest.mark.parametrize("method", get_methods())
def test_methods(method):
    slae = np.array([
        [2, 1, 4, 16],
        [3, 2, 1, 10],
        [1, 3, 3, 16],
    ], dtype="float64")
    solution = method(np.copy(slae))
    shape = slae.shape
    matrix = slae[:shape[0], :shape[0]]
    col = slae[:, [shape[0]]]
    assert np.allclose(matrix.dot(solution), col)
