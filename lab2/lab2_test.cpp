#include <lab2/lab2.hpp>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cmath>
#include <vector>

TEST_SUITE_BEGIN("function");

TEST_CASE("polynom_empty") {
    lab2::Polynom poly{};
    REQUIRE_EQ(poly(42.0), 0.0);
    REQUIRE_EQ(poly(0.0), 0.0);
    REQUIRE_EQ(poly(-1.0), 0.0);
}

TEST_CASE("polynom_linear") {
    lab2::Polynom poly{ 1.0, -2.0 };
    REQUIRE_EQ(poly(-1.0), 3.0);
    REQUIRE_EQ(poly(0.0), 1.0);
    REQUIRE_EQ(poly(1.0), -1.0);
}

TEST_CASE("polynom_square") {
    lab2::Polynom poly{ 1.0, 0.0, 1.0 };
    REQUIRE_EQ(poly(-1.0), 2.0);
    REQUIRE_EQ(poly(0.0), 1.0);
    REQUIRE_EQ(poly(1.0), 2.0);
}

TEST_CASE("polynom_derivative") {
    lab2::Polynom poly{ 1.0, 2.0, 3.0, 4.0 };
    auto derivative{ poly.derivative() };
    REQUIRE_EQ(derivative(-1.0), 8.0);
    REQUIRE_EQ(derivative(0.0), 2.0);
    REQUIRE_EQ(derivative(1.0), 20.0);
}

TEST_CASE("function_derivative") {
    lab2::Function exp{ static_cast<double(*)(double)>(std::exp) };
    auto derivative{ exp.derivative() };
    REQUIRE_EQ(exp(-1.0), doctest::Approx(derivative(-1.0)));
    REQUIRE_EQ(exp(0.0), doctest::Approx(derivative(0.0)));
    REQUIRE_EQ(exp(1.0), doctest::Approx(derivative(1.0)));

    lab2::Function sin{ static_cast<double(*)(double)>(std::sin) };
    auto cos{ sin.derivative() };
    REQUIRE_EQ(sin(-1.0) * sin(-1.0) + cos(-1.0) * cos(-1.0), doctest::Approx(1.0));
    REQUIRE_EQ(sin(0.0) * sin(0.0) + cos(0.0) * cos(0.0), doctest::Approx(1.0));
    REQUIRE_EQ(sin(1.0) * sin(1.0) + cos(1.0) * cos(1.0), doctest::Approx(1.0));
}

TEST_SUITE_END();

TEST_SUITE_BEGIN("method_newton");

TEST_CASE("method_newton") {
    const lab2::Polynom poly{ -7.0, 10.0, -8.0, -4.0, 3.0 };
    const double eps{ 0.001 };
    const std::vector<double> start{ 0.0, 1.0, -1.0, 10.0, -10.0 };
    for (const auto x0: start) {
        const double root{ lab2::methodNewton(poly, x0, eps) };
        CAPTURE(x0);
        CAPTURE(root);
        REQUIRE_EQ(poly(root), doctest::Approx(0.0).epsilon(eps));
    }
}

TEST_CASE("method_newton_2") {
    const lab2::Polynom poly{ 1.0, -1.0, 0.0, 1.0 };
    const double eps{ 0.001 };
    const double root{ lab2::methodNewton(poly, -2.0, eps) };
    CAPTURE(root);
    REQUIRE_EQ(poly(root), doctest::Approx(0.0).epsilon(eps));
}

TEST_SUITE_END();

TEST_SUITE_BEGIN("method_newton_simple");

TEST_CASE("method_newton_simple") {
    const lab2::Polynom poly{ 1.0, -1.0, 0.0, 1.0 };
    const double eps{ 0.001 };
    const double root{ lab2::methodNewtonSimple(poly, -2.0, eps) };
    CAPTURE(root);
    REQUIRE_EQ(poly(root), doctest::Approx(0.0).epsilon(0.01));
}

TEST_SUITE_END();

TEST_SUITE_BEGIN("method_secant");

TEST_CASE("method_secant") {
    const lab2::Polynom poly{ -7.0, 10.0, -8.0, -4.0, 3.0 };
    const double eps{ 0.001 };
    const double delta{ 0.1 };
    const std::vector<double> start{ 0.0, 4.0, 10.0 };
    for (const auto x0: start) {
        const double root{ lab2::methodSecant(poly, x0, eps, delta) };
        CAPTURE(x0);
        CAPTURE(root);
        REQUIRE_EQ(poly(root), doctest::Approx(0.0).epsilon(eps));
    }
}

TEST_CASE("method_secant_2") {
    const lab2::Polynom poly{ 1.0, -1.0, 0.0, 1.0 };
    const double eps{ 0.001 };
    const double delta{ 0.1 };
    const double root{ lab2::methodSecant(poly, -2.0, eps, delta) };
    CAPTURE(root);
    REQUIRE_EQ(poly(root), doctest::Approx(0.0).epsilon(eps));
}

TEST_SUITE_END();

TEST_SUITE_BEGIN("method_newton_broyden");

TEST_CASE("method_newton_broyden") {
    const auto func{ [](double x) { return 1.0/(1.0 + exp(-x)) - 0.5; } };
    const double eps{ 1.0e-6 };
    const std::vector<double> start{ 0.1, 1.0, 2.0 };
    for (const double x0: start) {
        const double root{ lab2::methodNewtonBroyden(func, x0, 1.0, eps) };
        CAPTURE(x0);
        CAPTURE(root);
        REQUIRE_EQ(func(root), doctest::Approx(0.0).epsilon(eps));
    }
}

TEST_CASE("method_newton_broyden_x0_2.5") {
    const auto func{ [](double x) { return 1.0/(1.0 + exp(-x)) - 0.5; } };
    const double eps{ 1.0e-3 };
    const double x0{ 2.5 };
    const double root{ lab2::methodNewtonBroyden(func, x0, 0.001, eps, 50) };
    CAPTURE(root);
    REQUIRE_EQ(func(root), doctest::Approx(0.0).epsilon(eps));
}

TEST_SUITE_END();

TEST_SUITE_BEGIN("roots_interval");

TEST_CASE("roots_interval") {
    const lab2::Polynom poly{ -7.0, 10.0, -8.0, -4.0, 3.0 };
    double left{};
    double right{};
    poly.rootsInterval(left, right);
    const double eps{ 1.0e-6 };
    const double root1{ lab2::methodNewton(poly, (left + right) / 2.0) };
    CAPTURE(root1);
    REQUIRE_GT(root1, 0.0);
    REQUIRE_GT(root1, left);
    REQUIRE_LT(root1, right);
    REQUIRE_EQ(poly(root1), doctest::Approx(0.0).epsilon(eps));
    const double root2{ lab2::methodNewton(poly, (-right + -left) / 2.0) };
    CAPTURE(root2);
    REQUIRE_LT(root2, 0.0);
    REQUIRE_GT(root2, -right);
    REQUIRE_LT(root2, -left);
    REQUIRE_EQ(poly(root2), doctest::Approx(0.0).epsilon(eps));
}

TEST_SUITE_END();
