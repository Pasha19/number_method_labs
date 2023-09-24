#ifndef NUMBER_METHODS_LABS_FUNCTION_HPP_
#define NUMBER_METHODS_LABS_FUNCTION_HPP_

#include <algorithm>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <vector>

namespace lab2 {

class FunctionHolder;

class FunctionInterface {
public:
    FunctionInterface() = default;
    FunctionInterface(const FunctionInterface&) = default;
    FunctionInterface(FunctionInterface&&) noexcept = default;
    FunctionInterface& operator=(const FunctionInterface&) = default;
    FunctionInterface& operator=(FunctionInterface&&) noexcept = default;
    virtual ~FunctionInterface() = default;

    virtual double operator()(double arg) const = 0;
    [[nodiscard]]
    virtual FunctionHolder derivative() const = 0;
};

class FunctionHolder: public FunctionInterface {
public:
    explicit FunctionHolder(std::unique_ptr<FunctionInterface>&& funcPtr)
        : FunctionInterface{}
        , funcPtr_{ std::move(funcPtr) }
    {}
    FunctionHolder(const FunctionHolder&) = delete;
    FunctionHolder(FunctionHolder&&) noexcept = default;
    FunctionHolder& operator=(const FunctionHolder&) = delete;
    FunctionHolder& operator=(FunctionHolder&&) noexcept = default;
    ~FunctionHolder() override = default;

    double operator()(const double arg) const override {
        return (*funcPtr_)(arg);
    }

    [[nodiscard]]
    FunctionHolder derivative() const override {
        return funcPtr_->derivative();
    }

private:
    std::unique_ptr<FunctionInterface> funcPtr_;
};

class Polynom: public FunctionInterface {
public:
    Polynom(std::initializer_list<double> coeff)
        : FunctionInterface{}
        , coeffs_{ coeff }
    {}
    template<typename ConstIterator>
    Polynom(ConstIterator cbegin, ConstIterator cend)
        : FunctionInterface{}
        , coeffs_{ cbegin, cend }
    {}
    Polynom(const Polynom&) = default;
    Polynom(Polynom&&) noexcept = default;
    Polynom& operator=(const Polynom&) = default;
    Polynom& operator=(Polynom&&) noexcept = default;
    ~Polynom() override = default;

    double operator()(double arg) const override;

    [[nodiscard]]
    FunctionHolder derivative() const override;

    void rootsInterval(double& left, double& right) const;

private:
    std::vector<double> coeffs_{};
};

class Function: public FunctionInterface {
public:
    explicit Function(std::function<double(double)> func, double h = 1.0e-6)
        : FunctionInterface{}
        , func_{ std::move(func) }
        , h_{ h }
    {}
    Function(const Function&) = default;
    Function(Function&&) noexcept = default;
    Function& operator=(const Function&) = default;
    Function& operator=(Function&&) noexcept = default;
    ~Function() override = default;

    double operator()(const double arg) const override { return func_(arg); }

    [[nodiscard]]
    FunctionHolder derivative() const override {
        return FunctionHolder{ std::make_unique<Function>(
            [this](const double arg) {
                return (this->func_(arg + this->h_) - this->func_(arg - this->h_)) / (2.0 * this->h_);
            },
            h_
        )};
    }

private:
    std::function<double(double)> func_{};
    double h_{};
};

}

#endif
