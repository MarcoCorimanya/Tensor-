#ifndef RELU_H
#define RELU_H

#include "TensorTransform.h"
#include <vector>

class ReLU : public TensorTransform {
public:
    Tensor apply(const Tensor& t) const override {
        std::vector<double> res;
        auto data = t.getData();
        auto shape = t.getShape();

        size_t size = 1;
        for (auto s : shape) size *= s;

        for (size_t i = 0; i < size; i++)
            res.push_back(data[i] > 0 ? data[i] : 0);

        return Tensor(shape, res);
    }
};

#endif