#ifndef SIGMOID_H
#define SIGMOID_H

#include "TensorTransform.h"
#include <cmath>

class Sigmoid : public TensorTransform {
public:
    Tensor apply(const Tensor& t) const override {
        std::vector<double> res;
        auto data = t.getData();
        auto shape = t.getShape();

        size_t size = 1;
        for (auto s : shape) size *= s;

        for (size_t i = 0; i < size; i++)
            res.push_back(1.0 / (1.0 + exp(-data[i])));

        return Tensor(shape, res);
    }
};

#endif