cmake_minimum_required(VERSION 3.14)
project(10sor)

set(CMAKE_CXX_STANDARD 14)

find_package(Boost REQUIRED)

add_executable(10sor main.cpp TensorView/Traits.h TensorView/TensorView.h TensorView/TensorViewFwd.h TensorView/Operations.h TensorView/Utils.h TensorView/TensorIO.h TensorView/Dims.h)

target_include_directories(10sor PRIVATE Boost)