#include <lab1/lab1.hpp>

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace lab1 {

constexpr bool is_number_eq_zero(const double number) {
    return std::abs(number) < std::numeric_limits<double>::epsilon();
}

double dih(
    double x0,
    double x1,
    const double eps,
    const unsigned int max_iterations,
    const std::function<double(double)> &function
) {
    if (eps <= 0) {
        throw std::runtime_error("eps <= 0");
    }
    if (x0 > x1) {
        throw std::runtime_error("x0 > x1");
    }
    double fx0{ function(x0) };
    if (is_number_eq_zero(fx0)) {
        return x0;
    }
    double fx1{ function(x1) };
    if (is_number_eq_zero(fx1)) {
        return x1;
    }
    if (fx0 * fx1 > 0) {
        throw std::runtime_error("same sign f(x0) and f(x1)");
    }
    double x2{ x0 + (x1 - x0) / 2.0 };
    double fx2{ function(x2) };
    for (unsigned int iteration{ 1 }; iteration < max_iterations; ++iteration) {
        if (is_number_eq_zero(fx2)) {
            return x2;
        }
        if (fx0 * fx2 > 0) {
            x0 = x2;
        } else {
            x1 = x2;
        }
        if (std::abs(x1 - x0) < eps) {
            return x2;
        }
        x2 = x0 + (x1 - x0) / 2.0;
        fx2 = function(x2);
    }
    return x2;
}

void start_approx(double &x0, double &x1) {
    std::random_device r{};
    std::default_random_engine engine{ r() };
    std::uniform_real_distribution<double> distr{ -1.0e3, 1.0e3 };
    x0 = distr(engine);
    x1 = distr(engine);
    if (x0 > x1) {
        std::swap(x0, x1);
    }
}

double simple_iteration(double x0, double eps, unsigned int max_iterations, double a) {
    if (eps <= 0) {
        throw std::runtime_error("eps <= 0");
    }
    if (x0 < 0) {
        throw std::runtime_error("x0 < 0");
    }
    if (a < 0) {
        throw std::runtime_error("a < 0");
    }
    double x1{ 0.5 * (x0 + a / x0) };
    double x2{ 0.5 * (x1 + a / x1) };
    for (int iteration{ 2 }; iteration < max_iterations; ++iteration) {
        if (is_number_eq_zero(x1 - x0)) {
            return x2;
        }
        const double q{ (x2 - x1) / (x1 - x0) };
        if (std::abs((x2 - x1) / (1 - q)) < eps) {
            return x2;
        }
        x0 = x1;
        x1 = x2;
        x2 = 0.5 * (x1 + a / x1);
    }
    return x2;
}

}
