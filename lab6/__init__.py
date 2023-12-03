from lab5.polynom import Polynom, PolynomRoot


class FuncNewton:
    def __init__(self, x: list[float], f: list[float]) -> None:
        assert len(x) == len(f) and len(x) > 1
        self.__x: list[float] = x[:]
        self.__f: list[float] = f[:]
        self.__diff = [list(self.__f[:])]
        self.__pol: Polynom = self.__build()

    def __build(self) -> Polynom:
        for i in range(len(self.__x) - 1):
            d = []
            for j in range(len(self.__x) - 1 - i):
                d.append((self.__diff[i][j + 1] - self.__diff[i][j]) / (self.__x[j + i + 1] - self.__x[j]))
            self.__diff.append(d)
        pol = Polynom([self.__f[0]])
        for i in range(1, len(self.__x)):
            pol += PolynomRoot(list(self.__x[:i]), self.__diff[i][0]).to_polynom()
        return pol

    def add(self, x: float, f: float) -> None:
        self.__x.append(x)
        self.__f.append(f)
        self.__diff[0].append(f)
        for i in range(1, len(self.__diff)):
            self.__diff[i].append(
                (self.__diff[i - 1][-1] - self.__diff[i - 1][-2]) / (self.__x[-1] - self.__x[-1 - i])
            )
        self.__diff.append([(self.__diff[-1][1] - self.__diff[-1][0]) / (self.__x[-1] - self.__x[0])])
        self.__pol += PolynomRoot(self.__x[:-1], self.__diff[-1][0]).to_polynom()

    def __call__(self, x: float) -> float:
        return self.__pol(x)

    def __str__(self) -> str:
        return str(self.__pol)


class FuncNewton1:
    def __init__(self, x: list[float], f: list[float]) -> None:
        assert len(x) == len(f) and len(x) > 1
        self.__x: list[float] = x[:]
        self.__f: list[float] = f[:]
        self.__pol: Polynom = self.__build()

    def __build(self) -> Polynom:
        diff = [list(self.__f[:])]
        for i in range(len(self.__x) - 1):
            d = []
            for j in range(len(self.__x) - 1 - i):
                d.append(diff[i][j + 1] - diff[i][j])
            diff.append(d)
        h = self.__x[1] - self.__x[0]
        q = PolynomRoot([self.__x[0]], 1.0 / h).to_polynom()
        qi = Polynom([1])
        fact = 1
        pol = Polynom([self.__f[0]])
        for i in range(1, len(self.__x)):
            fact *= i
            qi *= q + Polynom([1 - i])
            pol += qi * (diff[i][0] / fact)
        return pol

    def __call__(self, x: float) -> float:
        return self.__pol(x)

    def __str__(self) -> str:
        return str(self.__pol)


class FuncNewton2:
    def __init__(self, x: list[float], f: list[float]) -> None:
        assert len(x) == len(f) and len(x) > 1
        self.__x: list[float] = x[:]
        self.__f: list[float] = f[:]
        self.__pol: Polynom = self.__build()

    def __build(self) -> Polynom:
        diff = [list(self.__f[:])]
        for i in range(len(self.__x) - 1):
            d = []
            for j in range(len(self.__x) - 1 - i):
                d.append(diff[i][j + 1] - diff[i][j])
            diff.append(d)
        h = self.__x[1] - self.__x[0]
        q = PolynomRoot([self.__x[-1]], 1.0 / h).to_polynom()
        qi = Polynom([1])
        fact = 1
        pol = Polynom([self.__f[-1]])
        for i in range(1, len(self.__x)):
            fact *= i
            qi *= q + Polynom([i - 1])
            pol += qi * (diff[i][-1] / fact)
        return pol

    def __call__(self, x: float) -> float:
        return self.__pol(x)

    def __str__(self) -> str:
        return str(self.__pol)


def main():
    fn = FuncNewton([2, 3, 4, 5], [7, 5, 8, 7])
    x = 2.5
    print(f"N3(x) = {fn}")
    print(f"N3({x}) = {fn(x)}")
    fn.add(1, 5)
    print(f"add f(1) = 5")
    print(f"N4(x) = {fn}")
    print(f"N4({x}) = {fn(x)}")
    fn1 = FuncNewton1([2, 3, 4, 5], [7, 5, 8, 7])
    print(f"N3(1)(x) = {fn1}")
    print(f"N3(1)({x}) = {fn1(x)}")
    fn2 = FuncNewton2([2, 3, 4, 5], [7, 5, 8, 7])
    print(f"N3(2)(x) = {fn2}")
    print(f"N3(2)({x}) = {fn2(x)}")


if __name__ == "__main__":
    main()
