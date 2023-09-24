#include <lab2/lab2.hpp>

#include <cmath>

namespace lab2 {

double methodNewton(const FunctionInterface& func, double x0, const double eps, const unsigned int maxIterations) {
    const auto derivative{ func.derivative() };
    double x1{ x0 - func(x0) / derivative(x0) };
    for (unsigned int i{}; i < maxIterations && std::abs(x1 - x0) > eps; ++i) {
        x0 = x1;
        x1 = x0 - func(x0) / derivative(x0);
    }
    return x1;
}

double methodNewtonSimple(const FunctionInterface& func, double x0, const double eps, const unsigned int maxIterations) {
    const double dfx0{ func.derivative()(x0) };
    double x1{ x0 - func(x0) / dfx0 };
    for (unsigned int i{}; i < maxIterations && std::abs(x1 - x0) > eps; ++i) {
        x0 = x1;
        x1 = x0 - func(x0) / dfx0;
    }
    return x1;
}

double methodSecant(
    const FunctionInterface& func,
    double x0,
    const double eps,
    const double delta,
    const unsigned int maxIterations
) {
    double dfx{ (func(x0) - func(x0 - delta)) / delta };
    double x1{ x0 - func(x0) / dfx };
    for (unsigned int i{}; i < maxIterations && std::abs(x1 - x0) > eps; ++i) {
        dfx = (func(x1) - func(x0)) / (x1 - x0);
        x0 = x1;
        x1 = x0 - func(x0) / dfx;
    }
    return x1;
}

double methodNewtonBroyden(
    const FunctionInterface& func,
    double x0,
    double c,
    const double eps,
    const unsigned int maxIterations
) {
    const auto derivative{ func.derivative() };
    double x1{ x0 - c * func(x0) / derivative(x0) };
    double fx1{ func(x1) };
    double delta{ x1 - x0 };
    int cnt{};
    for (unsigned int i{}; i < maxIterations && std::abs(delta) > eps; ++i) {
        x0 = x1;
        double fx0{ fx1 };
        const double dfx0{ derivative(x0) };
        double newDelta{ c * fx0 / dfx0 };
        for (unsigned int j{}; j < maxIterations; ++j) {
            x1 = x0 - newDelta;
            fx1 = func(x1);
            if (std::abs(newDelta) < std::abs(delta) || std::abs(fx1) < std::abs(fx0)) {
                ++cnt;
                if (cnt == 5) {
                    cnt = 0;
                    c *= 2.0;
                } else {
                    break;
                }
            } else {
                c /= 2.0;
            }
            newDelta = c * fx0 / dfx0;
        }
        delta = newDelta;
    }
    return x1;
}

}
