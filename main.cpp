#include "Tensor.h"
#include "ReLU.h"
#include "Sigmoid.h"
#include <iostream>

int main() {

    Tensor input = Tensor::ones({1000,20,20});

    Tensor x = input.view({1000,400});

    Tensor W1 = Tensor::ones({400,100});
    Tensor b1 = Tensor::ones({1000,100});

    Tensor h = matmul(x,W1);
    h = h + b1;

    ReLU relu;
    h = h.apply(relu);

    Tensor W2 = Tensor::ones({100,10});
    Tensor b2 = Tensor::ones({1000,10});

    Tensor out = matmul(h,W2);
    out = out + b2;

    Sigmoid sig;
    out = out.apply(sig);

    std::cout << "Salida final:\n";
    out.print();
}