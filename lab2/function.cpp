#include <lab2/function.hpp>

namespace lab2 {

double Polynom::operator()(const double arg) const {
    double result{};
    double x{ 1.0 };
    for (const double c: coeffs_) {
        result += c * x;
        x *= arg;
    }
    return result;
}

FunctionHolder Polynom::derivative() const {
    auto derivative{ std::make_unique<Polynom, std::initializer_list<double>>({}) };
    if (coeffs_.size() < 2) {
        return FunctionHolder{ std::move(derivative) };
    }
    std::copy(coeffs_.cbegin() + 1, coeffs_.cend(), std::back_inserter(derivative->coeffs_));
    for (std::size_t i{ 0 }; i < derivative->coeffs_.size(); ++i) {
        derivative->coeffs_[i] *= static_cast<double>(i + 1);
    }
    return FunctionHolder{ std::move(derivative) };
}

void Polynom::rootsInterval(double &left, double &right) const {
    double a{ *std::max_element(coeffs_.begin(), coeffs_.end() - 1) };
    double b{ *std::max_element(coeffs_.begin() + 1, coeffs_.end()) };
    left = 1.0 / (1.0 + b / std::abs(coeffs_[0]));
    right = 1.0 + a / *(coeffs_.end() - 1);
}

}
