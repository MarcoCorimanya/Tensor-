cmake_minimum_required(VERSION 4.0)
project(TensorCC)

set(CMAKE_CXX_STANDARD 20)

add_executable(TensorCC main.cpp
        Tensor.cpp
        Tensor.h
        TensorTransform.h
        Sigmoid.h
        ReLU.h
)
