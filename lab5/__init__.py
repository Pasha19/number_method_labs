from lab5.polynom import Polynom, PolynomRoot


float4 = tuple[float, float, float, float]


class Func:
    def __init__(self, x: float4, f: float4):
        self.__x: float4 = x
        self.__f: float4 = f
        self.__pols: list[list[Polynom]] = []
        self.__build()

    def __build(self) -> None:
        for i in range(1, len(self.__x)):
            lst: list[Polynom] = []
            for j in range(len(self.__x) - i):
                lst.append(self.__build_range(j, j + i + 1))
            self.__pols.append(lst)
        self.__pols[1].insert(1, 0.5 * (self.__pols[1][0] + self.__pols[1][1]))

    def __build_range(self, begin: int, end: int) -> Polynom:
        p = PolynomRoot(list(self.__x[begin:end]))
        pol = Polynom([])
        for i in range(begin, end):
            d = 1.0
            for j in range(begin, end):
                if i == j:
                    continue
                d *= self.__x[i] - self.__x[j]
            tmp = p / PolynomRoot([self.__x[i]], d)
            tmp = tmp.to_polynom()
            tmp *= self.__f[i]
            pol += tmp
        return pol

    def __str__(self) -> str:
        result = ""
        for i in range(1, len(self.__x) - 1):
            result += f"n = {i}\n"
            for j in range(1, len(self.__x)):
                result += f"{self.__x[j - 1]} <= x <= {self.__x[j]} f(x) = "
                result += str(self.__pols[i - 1][j - 1])
                result += "\n"
        result += f"n = {len(self.__x) - 1}\n"
        result += f"f(x) = {self.__pols[-1][0]}\n"
        return result

    def __call__(self, x: float) -> list[float]:
        result: list[float] = []
        n = 0
        for i in range(1, len(self.__x)):
            n = i - 1
            if self.__x[i - 1] <= x <= self.__x[i]:
                break
        for i in range(len(self.__x) - 2):
            result.append(self.__pols[i][n](x))
        result.append(self.__pols[-1][0](x))
        return result


def main():
    f = Func((5, 6, 7, 8), (3, 0, 2, 1))
    print(f)
    x = 6.5
    result = f(x)
    for i in range(len(result)):
        print(f"n = {i + 1} f({x}) = {result[i]}")


if __name__ == "__main__":
    main()
