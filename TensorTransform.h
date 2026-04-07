#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "Tensor.h"

class TensorTransform {
public:
    virtual Tensor apply(const Tensor& t) const = 0;
    virtual ~TensorTransform() = default;
};

#endif