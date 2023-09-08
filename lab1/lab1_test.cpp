#include <lab1/lab1.hpp>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cmath>
#include <utility>
#include <vector>

TEST_SUITE_BEGIN("dih_tests");

TEST_CASE("dih_test_lambda_exceptions") {
    REQUIRE_THROWS_AS(lab1::dih(2.0, 1.0, 0.001, 20, [](double x) { return x; }), std::runtime_error&);
    REQUIRE_THROWS_AS(lab1::dih(1.0, 10.0, 0.001, 20, [](double x) { return x; }), std::runtime_error&);
    REQUIRE_THROWS_AS(lab1::dih(1.0, 2.0, -0.001, 20, [](double x) { return x; }), std::runtime_error&);
}

TEST_CASE("dih_test_lambda_x0_or_x1_zero") {
    const double eps{ 0.001 };
    const int max_iterations{ 20 };
    double root{ lab1::dih(0.0, 1.0, eps, max_iterations, [](double x) { return x; }) };
    REQUIRE(root == doctest::Approx(0.0).epsilon(eps));
    root = lab1::dih(-1.0, 0.0, eps, max_iterations, [](double x) { return x; });
    REQUIRE(root == doctest::Approx(0.0).epsilon(eps));
}

TEST_CASE("dih_test_cos") {
    const double eps{ 1.0e-6 };
    const int max_iterations{ 20 };
    double root{ lab1::dih(0.0, 3.0, eps, max_iterations, static_cast<double(*)(double)>(std::cos)) };
    REQUIRE(root == doctest::Approx(M_PI_2).epsilon(eps));
    root = lab1::dih(3.0, 6.0, eps, max_iterations, static_cast<double(*)(double)>(std::cos));
    REQUIRE(root == doctest::Approx(3 * M_PI_2).epsilon(eps));
}

TEST_CASE("dih_test_polynom") {
    const double eps{ 1.0e-6 };
    const int max_iterations{ 20 };
    // (x-5)^2*(x+1)*(x+4)*(x-2) = x^5 - 7x^4 - 11x^3 + 127x^2 -70x - 200
    double root{ lab1::dih(-10.0, -1.5, eps, max_iterations, { -200.0, -70.0, 127.0, -11.0, -7.0, 1.0 }) };
    REQUIRE(root == doctest::Approx(-4.0).epsilon(eps));
}

TEST_CASE("dih_test_start_approx") {
    const double eps{ 1.0e-6 };
    const int max_iterations{ 30 };
    for (int i{}; i < 100; ++i) {
        try {
            double x0{};
            double x1{};
            lab1::start_approx(x0, x1);
            double root{ lab1::dih(
                x0,
                x1,
                eps,
                max_iterations,
                [](double x) { return std::exp(x) - std::exp(1.5); }
            )};
            CAPTURE(root);
            CAPTURE(x0);
            CAPTURE(x1);
            CHECK(root == doctest::Approx(1.5).epsilon(eps));
            return;
        }
        catch (const std::runtime_error&) {}
    }
    WARN(false);
}

TEST_SUITE_END();

TEST_SUITE_BEGIN("simple_iteration_tests");

TEST_CASE("simple_iteration_tests_exceptions") {
    REQUIRE_THROWS_AS(lab1::simple_iteration(-1.0, 0.001, 20, 4.0), std::runtime_error&);
    REQUIRE_THROWS_AS(lab1::simple_iteration(1.0, 0.001, 20, -4.0), std::runtime_error&);
    REQUIRE_THROWS_AS(lab1::simple_iteration(1.0, -0.001, 20, 4.0), std::runtime_error&);
}

TEST_CASE("simple_iteration_test") {
    const double eps{ 1.0e-6 };
    const std::vector<std::pair<double, double>> test_data{
        {  2.0, 5.0 },
        { 10.0, 5.0 },
        {  2.0, 4.0 },
        {  1.0, 0.0 },
    };
    for (const auto& it : test_data) {
        const double x0{ it.first };
        const double a{ it.second };
        double sqroot{ lab1::simple_iteration(x0, eps, 20, a)};
        REQUIRE(sqroot * sqroot == doctest::Approx(a).epsilon(eps));
    }
}

TEST_SUITE_END();
