import pytest
from lab5.polynom import Polynom, PolynomRoot
from typing import Generator


def get_polynom_str() -> Generator[tuple[Polynom, str], None, None]:
    yield Polynom([]), "0"
    yield Polynom([42]), "42"
    yield Polynom([-42]), "-42"
    yield Polynom([0, 1]), "x"
    yield Polynom([0, 2]), "2x"
    yield Polynom([0, -1]), "-x"
    yield Polynom([0, -2]), "-2x"
    yield Polynom([-1, 1]), "x - 1"
    yield Polynom([-3, 2, 1]), "x^2 + 2x - 3"
    yield Polynom([0, 1, 0]), "x"
    yield Polynom([-2, 0, -5, 3, -1]), "-x^4 + 3x^3 - 5x^2 - 2"


@pytest.mark.parametrize("polynom_str", get_polynom_str())
def test_polynom_to_str(polynom_str: tuple[Polynom, str]):
    assert str(polynom_str[0]) == polynom_str[1]


def get_polynom_for_add() -> Generator[tuple[Polynom, Polynom, Polynom], None, None]:
    yield Polynom([]), Polynom([0]), Polynom([])
    yield Polynom([1]), Polynom([-1]), Polynom([])
    yield Polynom([1]), Polynom([0, 1]), Polynom([1, 1])


@pytest.mark.parametrize("polynoms", get_polynom_for_add())
def test_add(polynoms: tuple[Polynom, Polynom, Polynom]):
    assert polynoms[0] + polynoms[1] == polynoms[2]
    assert polynoms[1] + polynoms[0] == polynoms[2]


def test_mul_polynom_float():
    pol = Polynom([1, -2, 1])
    assert pol * 0 == Polynom([])
    assert 0 * pol == Polynom([])
    assert 2 * pol == Polynom([2, -4, 2])


def get_polynom_for_mul() -> Generator[tuple[Polynom, Polynom, Polynom], None, None]:
    yield Polynom([1]), Polynom([0]), Polynom([])
    yield Polynom([1]), Polynom([0, 1]), Polynom([0, 1])
    yield Polynom([1, 1]), Polynom([1, 1]), Polynom([1, 2, 1])
    yield Polynom([4, -3, 1]), Polynom([-1, 0, 1, 2]), Polynom([-4, 3, 3, 5, -5, 2])


@pytest.mark.parametrize("polynoms", get_polynom_for_mul())
def test_add(polynoms: tuple[Polynom, Polynom, Polynom]):
    assert polynoms[0] * polynoms[1] == polynoms[2]
    assert polynoms[1] * polynoms[0] == polynoms[2]


def test_polynom_root_to_polynom():
    pol = PolynomRoot([2, 3, 4])
    pol /= PolynomRoot([4], 0.5)
    assert pol.to_polynom() == Polynom([12, -10, 2])
