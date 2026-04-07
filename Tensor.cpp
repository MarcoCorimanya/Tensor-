#include "Tensor.h"
#include "TensorTransform.h"
#include <iostream>
#include <stdexcept>
#include <cmath>

size_t Tensor::compute_size(const std::vector<size_t>& shape) const {
    size_t s = 1;
    for (auto x : shape) s *= x;
    return s;
}

Tensor::Tensor(const std::vector<size_t>& shape, const std::vector<double>& values) {
    this->shape = shape;
    total_size = compute_size(shape);

    if (total_size != values.size())
        throw std::invalid_argument("Size mismatch");

    data = new double[total_size];
    for (size_t i = 0; i < total_size; i++)
        data[i] = values[i];
}

Tensor::Tensor(const Tensor& other) {
    shape = other.shape;
    total_size = other.total_size;
    data = new double[total_size];

    for (size_t i = 0; i < total_size; i++)
        data[i] = other.data[i];
}

Tensor::Tensor(Tensor&& other) noexcept {
    shape = other.shape;
    data = other.data;
    total_size = other.total_size;

    other.data = nullptr;
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) return *this;

    delete[] data;

    shape = other.shape;
    total_size = other.total_size;
    data = new double[total_size];

    for (size_t i = 0; i < total_size; i++)
        data[i] = other.data[i];

    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this == &other) return *this;

    delete[] data;

    shape = other.shape;
    data = other.data;
    total_size = other.total_size;

    other.data = nullptr;

    return *this;
}

Tensor::~Tensor() {
    delete[] data;
}

Tensor Tensor::zeros(const std::vector<size_t>& shape) {
    size_t size = 1;
    for (auto s : shape) size *= s;
    return Tensor(shape, std::vector<double>(size, 0));
}

Tensor Tensor::ones(const std::vector<size_t>& shape) {
    size_t size = 1;
    for (auto s : shape) size *= s;
    return Tensor(shape, std::vector<double>(size, 1));
}

Tensor Tensor::arange(int start, int end) {
    std::vector<double> v;
    for (int i = start; i < end; i++) v.push_back(i);
    return Tensor({v.size()}, v);
}

Tensor Tensor::operator+(const Tensor& other) const {
    if (total_size != other.total_size) throw std::invalid_argument("Dim error");
    std::vector<double> res(total_size);
    for (size_t i = 0; i < total_size; i++)
        res[i] = data[i] + other.data[i];
    return Tensor(shape, res);
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (total_size != other.total_size) throw std::invalid_argument("Dim error");
    std::vector<double> res(total_size);
    for (size_t i = 0; i < total_size; i++)
        res[i] = data[i] - other.data[i];
    return Tensor(shape, res);
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (total_size != other.total_size) throw std::invalid_argument("Dim error");
    std::vector<double> res(total_size);
    for (size_t i = 0; i < total_size; i++)
        res[i] = data[i] * other.data[i];
    return Tensor(shape, res);
}

Tensor Tensor::operator*(double scalar) const {
    std::vector<double> res(total_size);
    for (size_t i = 0; i < total_size; i++)
        res[i] = data[i] * scalar;
    return Tensor(shape, res);
}

Tensor Tensor::view(const std::vector<size_t>& new_shape) const {
    if (compute_size(new_shape) != total_size)
        throw std::invalid_argument("Invalid reshape");

    Tensor t = *this; // copia
    t.shape = new_shape;
    return t;
}

Tensor Tensor::unsqueeze(size_t dim) const {
    std::vector<size_t> new_shape = shape;
    new_shape.insert(new_shape.begin() + dim, 1);

    Tensor t = *this;
    t.shape = new_shape;
    return t;
}

Tensor Tensor::concat(const std::vector<Tensor>& tensors, size_t dim) {
    if (tensors.empty()) throw std::invalid_argument("Empty");

    std::vector<size_t> new_shape = tensors[0].shape;
    size_t total = 0;

    for (const auto& t : tensors)
        total += t.shape[dim];

    new_shape[dim] = total;

    size_t new_size = 1;
    for (auto s : new_shape) new_size *= s;

    std::vector<double> data_concat;
    for (const auto& t : tensors)
        for (size_t i = 0; i < t.total_size; i++)
            data_concat.push_back(t.data[i]);

    return Tensor(new_shape, data_concat);
}

Tensor Tensor::apply(const TensorTransform& transform) const {
    return transform.apply(*this);
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    auto sa = a.getShape();
    auto sb = b.getShape();

    if (sa.size() != 2 || sb.size() != 2 || sa[1] != sb[0])
        throw std::invalid_argument("matmul error");

    size_t m = sa[0], n = sa[1], p = sb[1];

    std::vector<double> res(m * p, 0);

    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < p; j++)
            for (size_t k = 0; k < n; k++)
                res[i*p + j] += a.data[i*n + k] * b.data[k*p + j];

    return Tensor({m, p}, res);
}

void Tensor::print() const {
    for (size_t i = 0; i < total_size; i++)
        std::cout << data[i] << " ";
    std::cout << "\n";
}

const std::vector<size_t>& Tensor::getShape() const { return shape; }
double* Tensor::getData() const { return data; }