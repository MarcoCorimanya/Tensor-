#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <initializer_list>

class TensorTransform;

class Tensor {
private:
    std::vector<size_t> shape;
    double* data;
    size_t total_size;

    size_t compute_size(const std::vector<size_t>& shape) const;

public:
    Tensor(const std::vector<size_t>& shape, const std::vector<double>& values);

    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;

    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;

    ~Tensor();

    static Tensor zeros(const std::vector<size_t>& shape);
    static Tensor ones(const std::vector<size_t>& shape);
    static Tensor arange(int start, int end);

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator*(double scalar) const;

    Tensor view(const std::vector<size_t>& new_shape) const;
    Tensor unsqueeze(size_t dim) const;

    static Tensor concat(const std::vector<Tensor>& tensors, size_t dim);

    Tensor apply(const TensorTransform& transform) const;

    friend Tensor matmul(const Tensor& a, const Tensor& b);

    void print() const;
    const std::vector<size_t>& getShape() const;
    double* getData() const;
};

#endif