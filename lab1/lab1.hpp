#ifndef NUMBER_METHODS_LABS_LAB1_HPP_
#define NUMBER_METHODS_LABS_LAB1_HPP_

#include <functional>
#include <iterator>
#include <initializer_list>

namespace lab1 {

double dih(
    double x0,
    double x1,
    double eps,
    unsigned int max_iterations,
    const std::function<double(double)> &function
);

/**
 * @param begin итератор, указывающий на начало списка коэффициентов полинома (при x^0)
 * @param end итератор, указывающий на конец списка коэффициентов полинома
 * Первым указывается коэффициент при нулевой степени, вторвм при первой и т.д
 * begin: a0, end: -> a0 + a1*x + a2*x^2 + ... + an*x^n
 */
template<typename ConstIterator>
inline double dih(double x0, double x1, double eps, unsigned int max_iterations, ConstIterator begin, ConstIterator end) {
    return dih(x0, x1, eps, max_iterations, [begin, end](double x) {
        double var{ 1.0 };
        double result{};
        for (auto it = begin; it != end; ++it) {
            result += var * *it;
            var *= x;
        }
        return result;
    });
}

/**
 * @param function коэффициенты полинома, начиная с коффициента при x^0
 * { a0, a1, a2, ..., an } -> a0 + a1*x + a2*x^2 + ... + an*x^n
 */
inline double dih(
    double x0,
    double x1,
    double eps,
    unsigned int max_iterations,
    const std::initializer_list<double> function
) {
    return dih(x0, x1, eps, max_iterations, function.begin(), function.end());
}

void start_approx(double &x0, double &x1);

double simple_iteration(double x0, double eps, unsigned int max_iterations, double a);

}

#endif
