#pragma once

#include "Tensor.h"
#include "TensorView.h"

namespace tensor_view {

template<class T1, class T2>
void softmax(const T1& src, T2& dst, size_t axis) {
    std::vector<size_t> size(src.shape(), src.shape() + T1::NumDims);
    size.erase(size.begin() + axis);
    Tensor<std::decay_t<typename T1::ValueType>, T1::NumDims - 1> tmp(size.data());
    src.max(tmp, axis);
    dst.assign_(src);
    dst -= tmp.unsqueeze(axis);
    dst.map_([](auto x) { return std::exp(x); });
    dst.sum(tmp, axis);
    dst /= tmp.unsqueeze(axis);
}

} // namespace tensor_view