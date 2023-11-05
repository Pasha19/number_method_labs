import math
from typing import Union


class Polynom:
    def __init__(self, a: list[float]):
        self.__a: list[float] = a.copy()
        self.__trim()

    def __copy__(self) -> "Polynom":
        return Polynom(self.__a)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Polynom):
            return False
        if len(self.__a) != len(other.__a):
            return False
        for i in range(len(self.__a)):
            if not math.isclose(self.__a[i], other.__a[i]):
                return False
        return True

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __str__(self) -> str:
        size = len(self.__a)
        if size == 0:
            return "0"
        if size == 1:
            return str(self.__a[0])
        result = ""
        if math.isclose(self.__a[-1], 1):
            result += "x"
        elif math.isclose(self.__a[-1], -1):
            result += "-x"
        else:
            result += f"{self.__a[-1]}x"
        n = size - 1
        if n > 1:
            result += f"^{n}"
        for ai in reversed(self.__a[:-1]):
            n -= 1
            if math.isclose(ai, 0):
                continue
            result += " + " if ai > 0 else " - "
            if n == 0:
                result += str(abs(ai))
            else:
                result += "x" if math.isclose(abs(ai), 1) else f"{abs(ai)}x"
                result += f"^{n}" if n > 1 else ""
        return result

    def __iadd__(self, other: "Polynom") -> "Polynom":
        min_size = min(len(self.__a), len(other.__a))
        for i in range(min_size):
            self.__a[i] += other.__a[i]
        if len(self.__a) < len(other.__a):
            for i in range(min_size, len(other.__a)):
                self.__a.append(other.__a[i])
        else:
            self.__trim()
        return self

    def __add__(self, other: "Polynom") -> "Polynom":
        return self.__copy__().__iadd__(other)

    def __imul__(self, other: Union["Polynom", int, float]) -> "Polynom":
        if type(other) in [int, float]:
            for i in range(len(self.__a)):
                self.__a[i] *= float(other)
            self.__trim()
            return self
        if len(self.__a) == 0 or len(other.__a) == 0:
            self.__a = []
            return self
        new_a = [0.0] * (len(self.__a) + len(other.__a) - 1)
        for i in range(len(other.__a)):
            for j in range(len(self.__a)):
                new_a[i + j] += other.__a[i] * self.__a[j]
        self.__a = new_a
        self.__trim()
        return self

    def __mul__(self, other: Union["Polynom", int, float]) -> "Polynom":
        return self.__copy__().__imul__(other)

    def __rmul__(self, other: int | float) -> "Polynom":
        return self.__copy__().__imul__(other)

    def __call__(self, x: float) -> float:
        t = 1.0
        result = 0.0
        for ai in self.__a:
            result += ai * t
            t *= x
        return result

    def __trim(self) -> None:
        to_remove = 0
        for ai in reversed(self.__a):
            if not math.isclose(ai, 0):
                break
            to_remove += 1
        if to_remove != 0:
            self.__a = self.__a[:-to_remove]


class RootNotFoundException(Exception):
    def __init__(self, root: float):
        self.root: float = root


class PolynomRoot:
    def __init__(self, x: list[float], k: float = 1.0):
        self.__k: float = k
        self.__x: list[float] = x.copy()

    def __copy__(self):
        return PolynomRoot(self.__x, self.__k)

    def __itruediv__(self, other: "PolynomRoot") -> "PolynomRoot":
        if self is other:
            self.__k = 1.0
            self.__x = []
            return self
        for x in other.__x:
            ind = None
            for i, xi in enumerate(self.__x):
                if math.isclose(x, xi):
                    ind = i
                    break
            if ind is None:
                raise RootNotFoundException(x)
            del self.__x[ind]
        self.__k /= other.__k
        return self

    def __truediv__(self, other: "PolynomRoot") -> "PolynomRoot":
        return self.__copy__().__itruediv__(other)

    def to_polynom(self) -> Polynom:
        result = Polynom([1.0])
        for x in self.__x:
            result *= Polynom([-x, 1.0])
        return result * self.__k
