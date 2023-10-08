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
    col = slae[:, -1]
    assert np.allclose(matrix.dot(solution), col)


def test_method_gauss_max_element():
    slae = np.array([
        [-3,  2.099, 6, 3.901],
        [10, -7,     0, 7],
        [ 5, -1,     5, 6],
    ], dtype="float64")
    solution = lab3.method_gauss_max_element(np.copy(slae))
    shape = slae.shape
    matrix = slae[:shape[0], :shape[0]]
    col = slae[:, -1]
    assert np.allclose(matrix.dot(solution), col)


def get_methods_iteration():
    yield lab3.method_simple_iteration
    yield lab3.method_seidel


@pytest.mark.parametrize("method", get_methods_iteration())
def test_methods_iteration(method):
    slae = np.array([
        [ 2,  2, 10, 14],
        [10,  1,  1, 12],
        [ 2, 10,  1, 13],
    ], dtype="float64")
    solution = method(np.copy(slae))
    shape = slae.shape
    matrix = slae[:shape[0], :shape[0]]
    col = slae[:, -1]
    assert np.allclose(matrix.dot(solution), col)


def test_method_lu():
    slae = np.array([
        [2, 1, 4, 16],
        [3, 2, 1, 10],
        [1, 3, 3, 16],
    ], dtype="float64")
    solution = lab3.method_lu(np.copy(slae))
    shape = slae.shape
    matrix = slae[:shape[0], :shape[0]]
    col = slae[:, -1]
    assert np.allclose(matrix.dot(solution), col)
