#ifndef NUMBER_METHODS_LABS_LAB2_HPP_
#define NUMBER_METHODS_LABS_LAB2_HPP_

#include <lab2/function.hpp>

namespace lab2 {

double methodNewton(const FunctionInterface& func, double x0, double eps = 1.0e-6, unsigned int maxIterations = 20);

double methodNewtonSimple(const FunctionInterface& func, double x0, double eps = 1.0e-6, unsigned int maxIterations = 20);

double methodSecant(
    const FunctionInterface& func,
    double x0,
    double eps = 1.0e-6,
    double delta = 0.1,
    unsigned int maxIterations = 20
);

double methodNewtonBroyden(
    const FunctionInterface& func,
    double x0,
    double c = 1.0,
    double eps = 1.0e-6,
    unsigned int maxIterations = 20
);

inline double methodNewton(
    std::function<double(double)> func,
    double x0,
    double eps = 1.0e-6,
    unsigned int maxIterations = 20
) {
    return methodNewton(Function{ std::move(func) }, x0, eps, maxIterations);
}

inline double methodNewton(
    std::initializer_list<double> coeff,
    double x0,
    double eps = 1.0e-6,
    unsigned int maxIterations = 20
) {
    return methodNewton(Polynom{ coeff }, x0, eps, maxIterations);
}

template<typename ConstIterator>
inline double methodNewton(
    ConstIterator cbegin,
    ConstIterator cend,
    double x0,
    double eps = 1.0e-6,
    unsigned int maxIterations = 20
) {
    return methodNewton(Polynom{ cbegin, cend }, x0, eps, maxIterations);
}

inline double methodNewtonSimple(
    std::function<double(double)> func,
    double x0,
    double eps = 1.0e-6,
    unsigned int maxIterations = 20
) {
    return methodNewtonSimple(Function{ std::move(func) }, x0, eps, maxIterations);
}

inline double methodNewtonSimple(
    std::initializer_list<double> coeff,
    double x0,
    double eps = 1.0e-6,
    unsigned int maxIterations = 20
) {
    return methodNewtonSimple(Polynom{ coeff }, x0, eps, maxIterations);
}

template<typename ConstIterator>
inline double methodNewtonSimple(
    ConstIterator cbegin,
    ConstIterator cend,
    double x0,
    double eps = 1.0e-6,
    unsigned int maxIterations = 20
) {
    return methodNewtonSimple(Polynom{ cbegin, cend }, x0, eps, maxIterations);
}

inline double methodSecant(
    std::function<double(double)> func,
    double x0,
    double eps = 1.0e-6,
    double delta = 0.1,
    unsigned int maxIterations = 20
) {
    return methodSecant(Function{ std::move(func) }, x0, eps, delta, maxIterations);
}

inline double methodSecant(
    std::initializer_list<double> coeff,
    double x0,
    double eps = 1.0e-6,
    double delta = 0.1,
    unsigned int maxIterations = 20
) {
    return methodSecant(Polynom{ coeff }, x0, eps, delta, maxIterations);
}

template<typename ConstIterator>
inline double methodSecant(
    ConstIterator cbegin,
    ConstIterator cend,
    double x0,
    double eps = 1.0e-6,
    double delta = 0.1,
    unsigned int maxIterations = 20
) {
    return methodSecant(Polynom{ cbegin, cend }, x0, eps, delta, maxIterations);
}

inline double methodNewtonBroyden(
    std::function<double(double)> func,
    double x0,
    double c = 1.0,
    double eps = 1.0e-6,
    unsigned int maxIterations = 20
) {
    return methodNewtonBroyden(Function{ std::move(func) }, x0, c, eps, maxIterations);
}

inline double methodNewtonBroyden(
    std::initializer_list<double> coeff,
    double x0,
    double c = 1.0,
    double eps = 1.0e-6,
    unsigned int maxIterations = 20
) {
    return methodNewtonBroyden(Polynom{ coeff }, x0, c, eps, maxIterations);
}

template<typename ConstIterator>
inline double methodNewtonBroyden(
    ConstIterator cbegin,
    ConstIterator cend,
    double x0,
    double c = 1.0,
    double eps = 1.0e-6,
    unsigned int maxIterations = 20
) {
    return methodNewtonBroyden(Polynom{ cbegin, cend }, x0, c, eps, maxIterations);
}

}

#endif
